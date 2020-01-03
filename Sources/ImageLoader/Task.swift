import Foundation

class Task<T> {
    private var value: T?
    private let lock = NSLock()
    
    init(handler: @escaping ()->T) {
        lock.lock()
        DispatchQueue.global().async {
            self.value = handler()
            self.lock.unlock()
        }
    }
    
    func get() -> T {
        lock.lock()
        defer { lock.unlock() }
        return value!
    }
}
