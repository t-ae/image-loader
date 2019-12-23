import CStbImage
import Foundation
import TensorFlow

public class ImageLoader {
    public struct Entry {
        public var url: URL
        public var label: Int32
        
        public init(url: URL, label: Int32) {
            self.url = url
            self.label = label
        }
    }
    
    public typealias Transform = (inout Tensor<Float>)->Void
    
    public var entries: [Entry]
    public var transforms: [Transform]
    private var rng: RandomNumberGenerator
    private var pointer: Int = 0
    
    /// Create `ImageLoader` with entries.
    public init(entries: [Entry],
                transforms: [Transform] = [],
                rng: RandomNumberGenerator = SystemRandomNumberGenerator()) {
        // Sort at first to ensure reproducibility.
        self.entries = entries.sorted { a, b in a.url.absoluteString < b.url.absoluteString }
        self.transforms = transforms
        self.rng = rng
    }
    
    /// Create `ImageLoader` with `directories`.
    public convenience init(directories: [(url: URL, label: Int32)],
                            extensions: [String] = ["bmp", "jpg", "jpeg", "png"],
                            transforms: [Transform] = [],
                            rng: RandomNumberGenerator = SystemRandomNumberGenerator()) throws {
        var entries = [Entry]()
        for directory in directories {
            let urls = FileManager.default.searchRecursively(directory: directory.url, extensions: extensions)
            entries.append(contentsOf: urls.map { Entry(url: $0, label: directory.label) })
        }
        self.init(entries: entries, transforms: transforms, rng: rng)
    }
    
    /// Create single class `ImageLoader` with image urls.
    /// All labels will be 0.
    public convenience init(urls: [URL],
                            transforms: [Transform] = [],
                            rng: RandomNumberGenerator = SystemRandomNumberGenerator()) {
        let entries = urls.map { Entry(url: $0, label: 0) }
        self.init(entries: entries, transforms: transforms, rng: rng)
    }
    
    /// Create single class `ImageLoader` with images in `directory`.
    /// All labels will be 0.
    public convenience init(directory: URL,
                            extensions: [String] = ["bmp", "jpg", "jpeg", "png"],
                            transforms: [Transform] = [],
                            rng: RandomNumberGenerator = SystemRandomNumberGenerator()) throws {
        let urls = FileManager.default.searchRecursively(directory: directory, extensions: extensions)
        self.init(urls: urls, transforms: transforms, rng: rng)
    }
    
    /// Shuffle `entries`.
    public func shuffle() {
        rng.shuffle(entries: &entries)
        pointer = 0
    }
    
    /// Generate next batch contains `size` images/labels.
    ///
    /// Values of `images` will be [0, 1] range if `trasnforms` doesn't change them.
    ///
    /// All images after `transfrom`s are applied must have same size/channels.
    public func nextBatch(size: Int) -> (images: Tensor<Float>, labels: Tensor<Int32>) {
        precondition(size > 0, "`size` must be greater than 0.")
        precondition(size < entries.count, "`entries` doesn't have `size` elements.")
        
        var end = pointer + size
        if end > entries.count {
            shuffle()
            end = size
        }
        
        let entries = self.entries[pointer..<end]
        pointer += size
        
        var images = [Tensor<Float>]()
        
        for entry in entries {
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
            
            images.append(image)
        }
        
        let imagesTensor = Tensor(stacking: images)
        let labelsTensor = Tensor(entries.map { $0.label })
        
        return (imagesTensor, labelsTensor)
    }
}

extension RandomNumberGenerator {
    mutating func shuffle(entries: inout [ImageLoader.Entry]) {
        entries.shuffle(using: &self)
    }
}

extension FileManager {
    func searchRecursively(directory: URL, extensions: [String]) -> [URL] {
        var urls = [URL]()
        
        guard let enumerator = enumerator(at: directory, includingPropertiesForKeys: nil) else {
            fatalError("Failed to get enumerator: \(directory)")
        }
        while let url = enumerator.nextObject() as? URL {
            if extensions.contains(url.pathExtension.lowercased()) {
                urls.append(url)
            }
        }
        
        return urls
    }
}
