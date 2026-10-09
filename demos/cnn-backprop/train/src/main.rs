//! Writes the digits the cnn-backprop demo trains on and tests with:
//! the first 600 digits of MNIST's training set into `../data/train.txt`
//! and the first 200 of its test set into `../data/test.txt`, one digit
//! per line: its label (0 to 9), then its 784 pixels (0 to 255, row by
//! row). MNIST (LeCun, Cortes and Burges) is fetched into work/mnist by
//! scripts/mnist.sh and is not committed; these few hundred digits are.
//! The network itself is trained by the demo, in X_eTaL.

const TRAIN: usize = 600;
const TEST: usize = 200;

/// An IDX file's payload after its header.
fn idx(path: &str, header: usize) -> Vec<u8> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e} (run scripts/mnist.sh first)"));
    b[header..].to_vec()
}

/// The first `n` digits of a set, one line each: the label, then the pixels.
fn lines(dir: &str, set: &str, n: usize) -> String {
    let img = idx(&format!("{dir}/{set}-images-idx3-ubyte"), 16);
    let lab = idx(&format!("{dir}/{set}-labels-idx1-ubyte"), 8);
    (0..n)
        .map(|i| {
            let px: Vec<String> = img[i * 784..(i + 1) * 784].iter().map(|p| p.to_string()).collect();
            format!("{} {}\n", lab[i], px.join(" "))
        })
        .collect()
}

fn main() {
    let here = env!("CARGO_MANIFEST_DIR");
    let mnist = format!("{here}/../../../work/mnist");
    let data = format!("{here}/../data");
    std::fs::create_dir_all(&data).expect("data/");
    std::fs::write(format!("{data}/train.txt"), lines(&mnist, "train", TRAIN)).expect("write data/train.txt");
    std::fs::write(format!("{data}/test.txt"), lines(&mnist, "t10k", TEST)).expect("write data/test.txt");
    eprintln!("wrote data/train.txt ({TRAIN} digits) and data/test.txt ({TEST})");
}
