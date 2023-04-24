# GPT-Encoder
Rust BPE Encoder Decoder for GPT-2 / GPT-3

##### This is rewrite of [openai's gpt-2 encoder](https://github.com/openai/gpt-2/blob/master/src/encoder.py) and [latitudegames's GPT-3-Encoder](https://github.com/latitudegames/GPT-3-Encoder) in rust

```rs
use gpt_encoder::Encoder;

fn main() {
    let mut encoder = Encoder::new();
    let encoded = encoder.encode("Hello, World".to_string());
    println!("{:?}", encoded); 
    // prints: [15496, 11, 2159]

    let decoded = encoder.decode(encoded);
    println!("{:?}", decoded); 
    // prints: "Hello, World"
}
```
