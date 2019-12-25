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
            let loader = try ImageLoader(directory: root, rng: XorshiftRandomNumberGenerator())
            
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
            ], rng: XorshiftRandomNumberGenerator())
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
        
        let rng = XorshiftRandomNumberGenerator()
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
        
        do {
            let loader = try ImageLoader(directories: [
                (cifar10, 0),
                (arbitrary, 1)
            ], transforms: [Transforms.resizeBilinear(width: 32, height: 64)], rng: XorshiftRandomNumberGenerator())
            
            for _ in 0..<10 {
                let (images, _) = loader.nextBatch(size: 13)
                XCTAssertEqual(images.shape, [13, 64, 32, 3])
            }
        }
        do {
            let loader = try ImageLoader(directories: [
                (cifar10, 0),
                (arbitrary, 1)
            ], transforms: [
                Transforms.resizeBilinear(width: 32, height: 64),
                Transforms.resizeBilinear(aspectFill: 20)
            ], rng: XorshiftRandomNumberGenerator())
            
            for _ in 0..<10 {
                let (images, _) = loader.nextBatch(size: 13)
                XCTAssertEqual(images.shape, [13, 40, 20, 3])
            }
        }
        do {
            let loader = try ImageLoader(directories: [
                (cifar10, 0),
                (arbitrary, 1)
            ], transforms: [
                Transforms.resizeBilinear(width: 32, height: 64),
                Transforms.centerCrop(width: 10, height: 20)
            ], rng: XorshiftRandomNumberGenerator())
            
            for _ in 0..<10 {
                let (images, _) = loader.nextBatch(size: 13)
                XCTAssertEqual(images.shape, [13, 20, 10, 3])
            }
        }
    }
    
    func testParallel() throws {
        let root = resourceRoot.appendingPathComponent("CIFAR10")
        
        let rng = XorshiftRandomNumberGenerator()
        let loader1 = try ImageLoader(directory: root, parallel: true, rng: rng)
        let loader2 = try ImageLoader(directory: root, parallel: false, rng: rng)
        
        for _ in 0..<100 {
            let (images1, _) = loader1.nextBatch(size: 13)
            let (images2, _) = loader2.nextBatch(size: 13)
            XCTAssertEqual(images1, images2)
        }
    }
    
    func testPerformanceSingleThread() throws {
        let root = resourceRoot.appendingPathComponent("CIFAR10")
        let loader = try ImageLoader(directory: root, parallel: false)
        
        measure  {
            for _ in 0..<100 {
                let (_, _) = loader.nextBatch(size: 32)
            }
        }
    }
    
    func testPerformanceParallel() throws {
        let root = resourceRoot.appendingPathComponent("CIFAR10")
        let loader = try ImageLoader(directory: root, parallel: true)
        
        measure  {
            for _ in 0..<100 {
                let (_, _) = loader.nextBatch(size: 32)
            }
        }
    }

    static var allTests = [
        ("testCIFAR10", testCIFAR10),
        ("testReproduction", testReproduction),
        ("testTransform", testTransform),
        ("testParallel", testParallel),
        ("testPerformanceSingleThread", testPerformanceSingleThread),
        ("testPerformanceParallel", testPerformanceParallel),
    ]
}
