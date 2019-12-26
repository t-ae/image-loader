# ImageLoader

Batch image loader for Swift for TensorFlow.

## Usage

```swift
import ImageLoader 

let class0Directory: URL = ...
let class1Directory: URL = ...
...

let loader = ImageLoader(directories: [
        (class0Directory, 0), // Specify directory and its label.
        (class1Directory, 1),
        ...
    ], 
    transforms: [
        Transforms.resize(width: 32, height: 32)
    ], // transforms are applied to each image.
    parallel: true, // If true, image loading is parallelized.
    rng: XorshiftRandomNumberGenerator()) // You can specify RNG for reproducibility.

for (images: Tensor<Float>, labels: Tensor<Int32>) in BatchImageSequence(loader: loader, batchSize: 32) {
    ...
}
```