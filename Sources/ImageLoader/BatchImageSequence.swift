public struct BatchImageSequence: Sequence, IteratorProtocol {
    public let loader: ImageLoader
    public let batchSize: Int
    public let infinite: Bool
    
    public var numberOfBatches: Int {
        loader.entries.count / batchSize
    }
    
    /// Create iterator.
    ///
    /// - Parameters:
    ///   - infinite: If false, `next` returns nil when epoch is end. Otherwise it shuffles `loader` and start iterating again.
    public init(loader: ImageLoader,
                batchSize: Int,
                infinite: Bool = false) {
        self.loader = loader
        self.batchSize = batchSize
        self.infinite = infinite
    }
    
    public mutating func next() -> MiniBatch? {
        if let next = loader.nextBatch(size: batchSize) {
            return next
        }
        
        // No more minibatches
        if infinite {
            loader.shuffleAndReset()
            return next()
        } else {
            return nil
        }
    }
}
