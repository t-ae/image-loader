import Foundation

extension RandomNumberGenerator {
    mutating func shuffle<T>(entries: inout [T]) {
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
