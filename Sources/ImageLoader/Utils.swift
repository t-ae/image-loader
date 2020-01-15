import Foundation

extension RandomNumberGenerator {
    mutating func shuffle<T>(entries: inout [T]) {
        entries.shuffle(using: &self)
    }
}

extension FileManager {
    func searchRecursively(directory: URL,
                           extensions: [String],
                           ignoreDotFiles: Bool) -> [URL] {
        var urls = OrderedURLs()
        
        guard let enumerator = enumerator(at: directory, includingPropertiesForKeys: nil) else {
            fatalError("Failed to get enumerator: \(directory)")
        }
        while let url = enumerator.nextObject() as? URL {
            guard url.lastPathComponent.first != "." else {
                continue
            }
            if extensions.contains(url.pathExtension.lowercased()) {
                urls.append(url)
            }
        }
        
        return urls.array
    }
}

struct OrderedURLs {
    /// Sorted array.
    private(set) var array: [URL]
    
    init() {
        array = []
    }
    
    mutating func reserveCapacity(minimumCapacity: Int) {
        array.reserveCapacity(minimumCapacity)
    }
    
    mutating func append(_ value: URL) {
        let index = binsearch(value)
        array.insert(value, at: index)
    }
    
    /// Find the index where `value` should be inserted.
    func binsearch(_ value: URL) -> Int {
        if array.endIndex == 0 {
            return 0
        }
        
        var left = array.startIndex
        var right = array.endIndex - 1
        
        if value.absoluteString < array[left].absoluteString {
            return 0
        }
        if value.absoluteString > array[right].absoluteString {
            return array.count
        }
        
        while true {
            let center = (left + right) / 2
            
            if center == left {
                return right
            }
            
            if value.absoluteString < array[center].absoluteString {
                right = center
            } else {
                left = center
            }
        }
    }
}
