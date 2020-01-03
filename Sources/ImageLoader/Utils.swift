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
        var urls = [URL]()
        
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
        
        return urls
    }
}
