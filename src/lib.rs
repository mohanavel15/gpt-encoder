//! # GPT-Encoder
//! Rust BPE Encoder Decoder for GPT-2 / GPT-3
//! 
//! This is rewrite of [openai's gpt-2 encoder](https://github.com/openai/gpt-2/blob/master/src/encoder.py) and [latitudegames's GPT-3-Encoder](https://github.com/latitudegames/GPT-3-Encoder) in rust
//! 
//! # Example
//! ```
//! use gpt_encoder::Encoder;
//! 
//! fn main() {
//!     let mut encoder = Encoder::new();
//!     let encoded = encoder.encode("Hello, World".to_string());
//!     println!("{:?}", encoded); 
//!     // prints: [15496, 11, 2159]
//! 
//!     let decoded = encoder.decode(encoded);
//!     println!("{:?}", decoded); 
//!     // prints: "Hello, World"
//! }
//! ```

use regex::Regex;
use serde_json::{from_str, Value};
use std::collections::{HashMap, HashSet};

fn bytes_to_unicode() -> HashMap<u8, char> {
    let mut bs: Vec<u8> = (b'!'..=b'~').collect();
    bs.extend(&(b'\xA1'..=b'\xAC').collect::<Vec<u8>>());
    bs.extend(&(b'\xAE'..=b'\xFF').collect::<Vec<u8>>());

    let mut cs: Vec<u32> = bs.iter().map(|&x| x as u32).collect();
    let mut n = 0;
    for b in 0..u8::MAX {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(2_u32.pow(8) + n);
            n += 1;
        }
    }

    let cs: Vec<char> = cs.iter().map(|&x| char::from_u32(x).unwrap()).collect();
    let result: HashMap<_, _> = bs.into_iter().zip(cs).collect();
    result
}

fn get_pairs(word: &Vec<String>) -> HashSet<(String, String)> {
    let mut pairs = HashSet::new();
    let mut prev_char = word.iter().next().unwrap();
    for char in word.iter().skip(1) {
        pairs.insert((prev_char.clone(), char.clone()));
        prev_char = char;
    }
    pairs
}

pub struct Encoder {
    pat: Regex,
    byte_encoder: HashMap<u8, char>,
    byte_decoder: HashMap<char, u8>,
    encoder: HashMap<String, u64>,
    decoder: HashMap<u64, String>,
    bpe_ranks: HashMap<(String, String), usize>,
    cache: HashMap<String, String>,
}

impl Encoder {
    /// To create new instance of Encoder
    pub fn new() -> Self {
        let pat = Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+").unwrap();
        let byte_encoder = bytes_to_unicode();
        let byte_decoder = byte_encoder.iter().map(|(&k, &v)| (v, k)).collect::<HashMap<_, _>>();

        let mut encoder = HashMap::new();
        let mut decoder = HashMap::new();

        let encoder_str = include_str!("encoder.json");
        let encoder_json: Value = from_str(&encoder_str).expect("Unable to parse JSON");
        for (key, value) in encoder_json.as_object().unwrap() {
            encoder.insert(key.clone(), value.as_u64().unwrap());
            decoder.insert(value.as_u64().unwrap(), key.clone());
        }

        let vocab_bpe = include_str!("vocab.bpe");
        let bpe_merges = vocab_bpe
            .split('\n')
            .skip(1)
            .take_while(|line| !line.is_empty())
            .map(|line| {
                let merge_str: Vec<&str> = line.split(' ').collect();
                (merge_str[0].to_owned(), merge_str[1].to_owned())
            })
            .collect::<Vec<(String, String)>>();

        let idx = 0..bpe_merges.len();
        let bpe_ranks = bpe_merges.into_iter().zip(idx).collect::<HashMap<_, _>>();

        let cache = HashMap::new();
        Self {
            pat,
            byte_encoder,
            byte_decoder,
            encoder,
            decoder,
            bpe_ranks,
            cache,
        }
    }

    fn bpe(&mut self, token: String) -> String {
        if let Some(cached_word) = self.cache.get(&token) {
            return cached_word.to_string();
        }
        
        let mut word = token.chars().map(|c| c.to_string()).collect::<Vec<_>>();
        let mut pairs = get_pairs(&word);
        if pairs.is_empty() {
            return token;
        }
    
        loop {
            let bigram = pairs
                .iter()
                .min_by_key(|pair| self.bpe_ranks.get(pair).unwrap_or(&usize::MAX))
                .unwrap();

            if !self.bpe_ranks.contains_key(bigram) {
                break;
            }
    
            let (first, second) = bigram;
            let mut new_word: Vec<String> = vec![];
            let mut i = 0;
            while i < word.len() {
                if let Some(j) = word[i..].iter().position(|c| c == first) {
                    new_word.extend(word[i..i+j].iter().map(|c| c.to_string()));
                    i += j;

                    if i < word.len() - 1 && &word[i] == first && &word[i + 1] == second {
                        new_word.push(first.to_string() + &second.to_string());
                        i += 2;
                    } else {
                        new_word.push(word[i].to_string());
                        i += 1;
                    }
                } else {
                    new_word.extend(word[i..].iter().map(|c| c.to_string()));
                    break;
                }
            }
    
            word = new_word;
            if word.len() == 1 {
                break;
            } else {
                pairs = get_pairs(&word);
            }
        }
    
        let word = word.join(" ");
        self.cache.insert(token, word.clone());
        word
    }
    
    /// # Example
    /// ```
    /// use gpt_encoder::Encoder;
    /// 
    /// let mut encoder = Encoder::new();
    /// let encoded = encoder.encode("Hello, World".to_string());
    /// assert_eq!(encoded, vec![15496, 11, 2159]);
    /// ```
    pub fn encode(&mut self, text: String) -> Vec<u64> {
        let mut bpe_tokens = vec![];

        let matches: Vec<&str> = self.pat
            .find_iter(text.as_str())
            .map(|m| m.as_str())
            .filter(|s| !s.is_empty())
            .collect();

        for token in matches {
            let token = token
                .bytes()
                .map(|x| self.byte_encoder.get(&x).unwrap().to_string())
                .collect::<Vec<_>>()
                .join("");
            
            let mut new_tokens = self.bpe(token)
                .split(' ')
                .map(|x| self.encoder.get(&x.to_string()))
                .filter(|x| x.is_some())
                .map(|x| x.unwrap().clone())
                .collect::<Vec<_>>();
            bpe_tokens.append(&mut new_tokens);
        }
        bpe_tokens
    }

    /// # Example
    /// ```
    /// use gpt_encoder::Encoder;
    /// 
    /// let encoder = Encoder::new();
    /// let decoded = encoder.decode(vec![15496, 11, 2159]);
    /// assert_eq!(decoded, "Hello, World".to_string());
    /// ```
    pub fn decode(&self, token: Vec<u64>) -> String {
        let text: String = token.iter().map(|t| self.decoder.get(t).unwrap().clone()).collect::<Vec<_>>().join("");
        let text: Vec<u8> = text.chars().map(|c| self.byte_decoder.get(&c).unwrap().clone()).collect::<Vec<_>>();
        String::from_utf8_lossy(&text).to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() -> Result<(), String> {
        let mut encoder = Encoder::new();

        test_encoder(&mut encoder,"Space", " ".to_string(), vec![220])?;
        test_encoder(&mut encoder,"Tab", "\t".to_string(), vec![197])?;
        test_encoder(&mut encoder,"Simple text", "This is some text".to_string(), vec![1212, 318, 617, 2420])?;
        test_encoder(&mut encoder,"indivisible", "indivisible".to_string(), vec![521, 452, 12843])?;
        test_encoder(&mut encoder,"emojis", "hello 👋 world 🌍".to_string(), vec![31373, 50169, 233, 995, 12520, 234, 235])?;
        test_encoder(&mut encoder,"properties of Object", "toString constructor hasOwnProperty valueOf".to_string(), vec![1462, 10100, 23772, 468, 23858, 21746, 1988, 5189])?;

        Ok(())
    }

    fn test_encoder(encoder: &mut Encoder, title: &str, text: String, expected: Vec<u64>) -> Result<(), String> {
        let encoded = encoder.encode(text.clone());
        if encoded != expected {
            return Err(format!("{}: encoded output did not match the expected output", title))
        }
    
        let decoded = encoder.decode(encoded);
        if decoded != text {
            return Err(format!("{}: decoded output did not match the input text", title))
        }
    
        Ok(())
    }
}
