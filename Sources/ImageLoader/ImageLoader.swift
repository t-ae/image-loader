import CStbImage
import Foundation
import TensorFlow

public typealias MiniBatch = (images: Tensor<Float>, labels: Tensor<Int32>)

/// Image loader.
///
/// For for-loop iteration, use `BatchImageSequence`.
public class ImageLoader {
    /// Entries of dataset.
    public var entries: [Entry]
    
    /// Transformations which will be applied to loaded images individually.
    public var transforms: [Transform]
    
    public var parallel: Bool
    
    /// Random number generator used for shuffling entries.
    private var rng: RandomNumberGenerator
    
    /// Pointing where next batch will start.
    private var pointer: Int = 0
    
    /// For time profiling.
    public var onNextBatchEnd: (TimeInterval)->Void = { _ in }
    
    /// Create `ImageLoader` with entries.
    public init(entries: [Entry],
                transforms: [Transform] = [],
                parallel: Bool = false,
                rng: RandomNumberGenerator = SystemRandomNumberGenerator()) {
        // Sort at first to ensure reproducibility.
        self.entries = entries.sorted { a, b in a.url.absoluteString < b.url.absoluteString }
        self.transforms = transforms
        self.parallel = parallel
        self.rng = rng
    }
    
    /// Create `ImageLoader` with `directories`.
    public convenience init(directories: [(url: URL, label: Int32)],
                            extensions: [String] = ["bmp", "jpg", "jpeg", "png"],
                            transforms: [Transform] = [],
                            parallel: Bool = false,
                            rng: RandomNumberGenerator = SystemRandomNumberGenerator()) throws {
        var entries = [Entry]()
        for directory in directories {
            let urls = FileManager.default.searchRecursively(directory: directory.url, extensions: extensions)
            entries.append(contentsOf: urls.map { Entry(url: $0, label: directory.label) })
        }
        self.init(entries: entries, transforms: transforms, parallel: parallel, rng: rng)
    }
    
    /// Create single class `ImageLoader` with image urls.
    /// All labels will be 0.
    public convenience init(urls: [URL],
                            transforms: [Transform] = [],
                            parallel: Bool = false,
                            rng: RandomNumberGenerator = SystemRandomNumberGenerator()) {
        let entries = urls.map { Entry(url: $0, label: 0) }
        self.init(entries: entries, transforms: transforms, parallel: parallel, rng: rng)
    }
    
    /// Create single class `ImageLoader` with images in `directory`.
    /// All labels will be 0.
    public convenience init(directory: URL,
                            extensions: [String] = ["bmp", "jpg", "jpeg", "png"],
                            transforms: [Transform] = [],
                            parallel: Bool = false,
                            rng: RandomNumberGenerator = SystemRandomNumberGenerator()) throws {
        let urls = FileManager.default.searchRecursively(directory: directory, extensions: extensions)
        self.init(urls: urls, transforms: transforms, parallel: parallel, rng: rng)
    }
    
    /// Shuffle `entries`.
    public func shuffleAndReset() {
        rng.shuffle(entries: &entries)
        pointer = 0
    }
    
    /// Generate next batch contains `size` images/labels. Returns nil when there's no more batches.
    ///
    /// Values of `images` will be [0, 1] range if `trasnforms` doesn't change them.
    ///
    /// All images after `transfrom`s are applied must have same size/channels.
    public func nextBatch(size: Int) -> MiniBatch? {
        let startTime = Date()
        defer { onNextBatchEnd(Date().timeIntervalSince(startTime)) }
        
        precondition(size > 0, "`size` must be greater than 0.")
        precondition(size < entries.count, "`entries` doesn't have `size` elements.")
        
        let end = pointer + size
        guard end <= entries.count else {
            return nil
        }
        
        let entries = self.entries[pointer..<end]
        pointer += size
        
        var images = [Tensor<Float>](repeating: Tensor<Float>.zero, count: size)
        func loadImage(entry: Entry, index: Int) {
            var (width, height, channels): (Int32, Int32, Int32) = (0, 0, 0)
            guard let data = load_image(entry.url.path, &width, &height, &channels, 0) else {
                fatalError("Failed to load image: \(entry.url.path)")
            }
            defer { free_image(data) }
            
            let bp = UnsafeBufferPointer<UInt8>(start: data, count: Int(width*height*channels))
            var image = Tensor(bp.map { Float($0) / 255 })
                .reshaped(to: [Int(height), Int(width), Int(channels)])
            
            for transform in transforms {
                transform(&image)
            }
            precondition(image.shape.count == 3, "Rank of `image` is not 3 after `trasnform`s are applied.")
            
            images[index] = image
        }
        
        if parallel {
            DispatchQueue.concurrentPerform(iterations: entries.count) { i in
                let entry = entries[entries.startIndex + i]
                loadImage(entry: entry, index: i)
            }
        } else {
            for (i, entry) in entries.enumerated() {
                loadImage(entry: entry, index: i)
            }
        }
        
        let imagesTensor = Tensor(stacking: images)
        let labelsTensor = Tensor(entries.map { $0.label })
        
        return (imagesTensor, labelsTensor)
    }
}
