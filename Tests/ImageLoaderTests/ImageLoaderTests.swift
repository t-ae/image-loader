import XCTest
import TensorFlow
import ImageLoader

final class ImageLoaderTests: XCTestCase {
    
    let resourceRoot = URL(fileURLWithPath: #file)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .appendingPathComponent("Resources")
    
    func testSorted() {
        let root = resourceRoot.appendingPathComponent("CIFAR10")
        let entries = [Entry](directory: root)
        
        let urls = entries.map { $0.url }
        XCTAssertEqual(urls, urls.sorted { a, b in a.absoluteString < b.absoluteString })
    }
    
    func testCIFAR10() throws {
        let root = resourceRoot.appendingPathComponent("CIFAR10")
        do { // as single class
            let entries = [Entry](directory: root)
            let loader = ImageLoader(entries: entries, rng: XorshiftRandomNumberGenerator())
            loader.shuffle()
            
            var lastImages: Tensor<Float> = [0]
            let zeroLabels = Tensor<Int32>(zeros: [13])
            for (images, labels) in loader.iterator(batchSize: 13) {
                XCTAssertEqual(images.shape, [13, 32, 32, 3])
                XCTAssertNotEqual(images, lastImages)
                XCTAssertEqual(labels, zeroLabels)
                lastImages = images
            }
        }
        do { // multi class
            let entries = [Entry](directories: [
                (root.appendingPathComponent("airplane"), 0),
                (root.appendingPathComponent("automobile"), 1),
                (root.appendingPathComponent("bird"), 2),
                (root.appendingPathComponent("cat"), 3),
            ])
            let loader = ImageLoader(entries: entries, rng: XorshiftRandomNumberGenerator())
            loader.shuffle()
            
            var lastImages: Tensor<Float> = [0]
            var lastLabels: Tensor<Int32> = [0]
            for (images, labels) in loader.iterator(batchSize: 13) {
                XCTAssertEqual(images.shape, [13, 32, 32, 3])
                XCTAssertNotEqual(images, lastImages)
                XCTAssertNotEqual(labels, lastLabels)
                (lastImages, lastLabels) = (images, labels)
            }
        }
    }
    
    func testReproduction() throws {
        let root = resourceRoot.appendingPathComponent("CIFAR10")
        let entries = [Entry](directory: root)
        
        let rng = XorshiftRandomNumberGenerator()
        let loader1 = ImageLoader(entries: entries, rng: rng)
        let loader2 = ImageLoader(entries: entries, rng: rng)
        
        let ziploader = zip(loader1.iterator(batchSize: 13), loader2.iterator(batchSize: 13))
        
        for ((images1, _), (images2, _)) in ziploader {
            XCTAssertEqual(images1, images2)
        }
    }
    
    func testTransform() throws {
        let cifar10 = resourceRoot.appendingPathComponent("CIFAR10")
        let arbitrary = resourceRoot.appendingPathComponent("arbitrary_size")
        
        do {
            let entries = [Entry](directories: [
                (cifar10, 0),
                (arbitrary, 1)
            ])
            
            let loader = ImageLoader(entries: entries, transforms: [
                Transforms.resize(.bilinear, width: 32, height: 64)
            ], rng: XorshiftRandomNumberGenerator())
            
            for (images, _) in loader.iterator(batchSize: 13) {
                XCTAssertEqual(images.shape, [13, 64, 32, 3])
            }
        }
        do {
            let entries = [Entry](directories: [
                (cifar10, 0),
                (arbitrary, 1)
            ])
            
            let loader = ImageLoader(entries: entries, transforms: [
                Transforms.resize(.area, width: 32, height: 64),
                Transforms.resize(.nearestNeighbor, aspectFill: 20)
            ], rng: XorshiftRandomNumberGenerator())
            
            for (images, _) in loader.iterator(batchSize: 13) {
                XCTAssertEqual(images.shape, [13, 40, 20, 3])
            }
        }
        do {
            let entries = [Entry](directories: [
                (cifar10, 0),
                (arbitrary, 1)
            ])
            
            let loader = ImageLoader(entries: entries, transforms: [
                Transforms.resize(.bicubic, width: 32, height: 64),
                Transforms.centerCrop(width: 10, height: 20)
            ], rng: XorshiftRandomNumberGenerator())
            
            for (images, _) in loader.iterator(batchSize: 13) {
                XCTAssertEqual(images.shape, [13, 20, 10, 3])
            }
        }
    }
    
    func testNextTime() {
        let root = resourceRoot.appendingPathComponent("CIFAR10")
        let entries = [Entry](directory: root)
        let loader = ImageLoader(entries: entries)
        let iter = loader.iterator(batchSize: 32)
        
        // First next is almost synchronus
        let first = Date()
        _ = iter.next()
        let firstTime = Date().timeIntervalSince(first)
        
        // Training step
        sleep(1)
        
        // Second next is already loaded while training step
        let second = Date()
        _ = iter.next()
        let secondTime = Date().timeIntervalSince(second)
        
        // Second is faster than first
        XCTAssertLessThan(secondTime, firstTime * 0.1)
    }
    
    func testPerformance() throws {
        let root = resourceRoot.appendingPathComponent("CIFAR10")
        let entries = [Entry](directory: root)
        let loader = ImageLoader(entries: entries)
        
        measure  {
            for _ in 0..<100 {
                for batch in loader.iterator(batchSize: 32) {
                    _ = batch
                }
            }
        }
    }

    static var allTests = [
        ("testSorted", testSorted),
        ("testCIFAR10", testCIFAR10),
        ("testReproduction", testReproduction),
        ("testTransform", testTransform),
        ("testNextTime", testNextTime),
        ("testPerformance", testPerformance),
    ]
}
