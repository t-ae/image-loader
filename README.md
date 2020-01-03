# ImageLoader

Batch image loader for Swift for TensorFlow.

## Usage

```swift
import ImageLoader 

let class0Directory: URL = ...
let class1Directory: URL = ...
...

let entries = [Entry](directories: [
    (class0Directory, 0), // Specify directory and its label.
    (class1Directory, 1),
    ...
])

let loader = ImageLoader(
    entries: entries, 
    transforms: [ // transforms are applied to each image.
        Transforms.resize(width: 32, height: 32)
    ],
    rng: XorshiftRandomNumberGenerator()) // You can specify RNG for reproducibility.
)

for (images: Tensor<Float>, labels: Tensor<Int32>) in loader.iterator(batchSize: 32) {
    // While training step, iterator creates next result of `next` on background.
    // It improves training time.
    ...
}
```