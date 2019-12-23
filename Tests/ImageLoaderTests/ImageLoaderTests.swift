import XCTest
import TensorFlow
import ImageLoader

final class ImageLoaderTests: XCTestCase {
    
    let resourceRoot = URL(fileURLWithPath: #file)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .appendingPathComponent("Resources")
    
    func testCIFAR10() throws {
        let root = resourceRoot.appendingPathComponent("CIFAR10")
        do { // as single class
            let loader = try ImageLoader(directory: root, rng: RNG())
            
            var lastImages: Tensor<Float> = [0]
            let zeroLabels = Tensor<Int32>(zeros: [13])
            for _ in 0..<100 {
                let (images, labels) = loader.nextBatch(size: 13)
                XCTAssertEqual(images.shape, [13, 32, 32, 3])
                XCTAssertNotEqual(images, lastImages)
                XCTAssertEqual(labels, zeroLabels)
                lastImages = images
            }
        }
        do { // multi class
            let loader = try ImageLoader(directories: [
                (root.appendingPathComponent("airplane"), 0),
                (root.appendingPathComponent("automobile"), 1),
                (root.appendingPathComponent("bird"), 2),
                (root.appendingPathComponent("cat"), 3),
            ], rng: RNG())
            var lastImages: Tensor<Float> = [0]
            var lastLabels: Tensor<Int32> = [0]
            for _ in 0..<100 {
                let (images, labels) = loader.nextBatch(size: 13)
                XCTAssertEqual(images.shape, [13, 32, 32, 3])
                XCTAssertNotEqual(images, lastImages)
                XCTAssertNotEqual(labels, lastLabels)
                (lastImages, lastLabels) = (images, labels)
            }
        }
    }
    
    func testReproduction() throws {
        let root = resourceRoot.appendingPathComponent("CIFAR10")
        
        let rng = RNG()
        let loader1 = try ImageLoader(directory: root, rng: rng)
        let loader2 = try ImageLoader(directory: root, rng: rng)
        
        for _ in 0..<100 {
            let (images1, _) = loader1.nextBatch(size: 13)
            let (images2, _) = loader2.nextBatch(size: 13)
            XCTAssertEqual(images1, images2)
        }
    }
    
    func testTransform() throws {
        let cifar10 = resourceRoot.appendingPathComponent("CIFAR10")
        let arbitrary = resourceRoot.appendingPathComponent("arbitrary_size")
        
        let loader = try ImageLoader(directories: [
            (cifar10, 0),
            (arbitrary, 1)
        ], transforms: [Transforms.resizeBilinear(width: 32, height: 32)], rng: RNG())
        
        for _ in 0..<100 {
            let (images, _) = loader.nextBatch(size: 13)
            XCTAssertEqual(images.shape, [13, 32, 32, 3])
        }
    }

    static var allTests = [
        ("testCIFAR10", testCIFAR10),
        ("testReproduction", testReproduction),
        ("testTransform", testTransform)
    ]
}

/// Xorshift64
struct RNG: RandomNumberGenerator {
    var x: UInt64 = 88172645463325252;
    mutating func next() -> UInt64 {
        x = x ^ (x << 7)
        x = x ^ (x >> 9)
        return x
    }
}
