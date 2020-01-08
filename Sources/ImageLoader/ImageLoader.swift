import Foundation
import TensorFlow

public typealias MiniBatch = (images: Tensor<Float>, labels: Tensor<Int32>)

public class ImageLoader {
    public private(set) var entries: [Entry]
    public private(set) var transforms: [Transform]
    public var rng: RandomNumberGenerator
    
    public init(entries: [Entry],
                transforms: [Transform] = [],
                rng: RandomNumberGenerator = SystemRandomNumberGenerator()) {
        
        self.entries = entries
        self.transforms = transforms
        self.rng = rng
    }
    
    /// Create iterator.
    public func iterator(batchSize: Int) -> ImageIterator {
        precondition(entries.count >= batchSize, "`entries` doesn't have `batchSize` elements.")
        return ImageIterator(entries: entries, batchSize: batchSize, transforms: transforms)
    }
    
    /// Shuffle `entries` with `rng`.
    public func shuffle() {
        rng.shuffle(entries: &entries)
    }
}

public class ImageIterator: Sequence, IteratorProtocol {
    private let entries: [Entry]
    private let transforms: [Transform]
    private let batchSize: Int
    
    private var pointer = 0
    private var nextMiniBatch: Task<MiniBatch?>? = nil
    private let syncQueue = DispatchQueue(label: "ImageIterator")
    
    init(entries: [Entry],
         batchSize: Int,
         transforms: [Transform] = []) {
        self.entries = entries
        self.transforms = transforms
        self.batchSize = batchSize
    }
    
    func fireNextTask() {
        let end = pointer + batchSize
        if entries.count < end {
            // Epoch end
            nextMiniBatch = Task { nil }
            return
        }
        let slice = entries[pointer..<end]
        nextMiniBatch = Task {
            self.createMinibatch(entries: slice)
        }
        pointer += batchSize
    }
    
    func createMinibatch(entries: ArraySlice<Entry>) -> MiniBatch {
        var images = [Tensor<Float>](repeating: Tensor<Float>.zero, count: entries.count)
        
        DispatchQueue.concurrentPerform(iterations: entries.count) { i in
            withDevice(.cpu) {
                let entry = entries.dropFirst(i).first!
                var image = loadImage(url: entry.url)
                for t in transforms {
                    t(&image)
                }
                precondition(image.shape.count == 3,
                             "Rank of `image` is not 3 after `trasnform`s are applied.")
                syncQueue.sync {
                    images[i] = image
                }
            }
        }
        
        let imagesTensor = Tensor(stacking: images)
        let labelsTensor = Tensor(entries.map { $0.label })
        
        return (imagesTensor, labelsTensor)
    }
    
    public func next() -> MiniBatch? {
        if nextMiniBatch == nil {
            // First call
            fireNextTask()
        }
        let minibatch = nextMiniBatch!.get()
        fireNextTask()
        return minibatch
    }
}
