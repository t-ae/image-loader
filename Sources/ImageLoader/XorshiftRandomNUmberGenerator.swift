import Foundation

/// Xorshift64 pseudorandom number generator.
public struct XorshifrRandomNumberGenerator: RandomNumberGenerator {
    private var x: UInt64
    
    public init(seed: UInt64 = 88172645463325252) {
        self.x = seed
    }
    
    public mutating func next() -> UInt64 {
        x = x ^ (x << 7)
        x = x ^ (x >> 9)
        return x
    }
}
