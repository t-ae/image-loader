import Foundation

public struct Entry {
    public var url: URL
    public var label: Int32
    
    public init(url: URL, label: Int32) {
        self.url = url
        self.label = label
    }
}

extension Array where Element == Entry {
    public init(directories: [(url: URL, label: Int32)],
                extensions: [String] = ["bmp", "png", "jpg", "jpeg"]) {
        self.init()
        self.reserveCapacity(directories.count * 100)
        
        for directory in directories {
            let urls = FileManager.default.searchRecursively(directory: directory.url,
                                                             extensions: extensions,
                                                             ignoreDotFiles: true)
            self.append(contentsOf: urls.map { Entry(url: $0, label: directory.label) })
        }
    }
    
    /// Create single class entries.
    /// All labels will be 0.
    public init(urls: [URL]) {
        self = urls.map { Entry(url: $0, label: 0) }
    }
    
    /// Create single class entries with images in `directory`.
    /// All labels will be 0.
    public init(directory: URL,
                extensions: [String] = ["bmp", "png", "jpg", "jpeg"]) {
        let urls = FileManager.default.searchRecursively(directory: directory,
                                                         extensions: extensions,
                                                         ignoreDotFiles: true)
        self.init(urls: urls)
    }
}
