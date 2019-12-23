# ImageLoader

Batch image loader for Swift for TensorFlow.

## Usage

```swift
import ImageLoader 

let class0Directory: URL = ...
let class1Directory: URL = ...
...

let loader = ImageLoader(directories: [
    (class0Directory, 0),
    (class1Directory, 1),
    ...
])

for _ in 0..<numSteps {
    let (images, labels) = loader.nextBatch(size: batchSize)
    ...
}
```