import Foundation
import TensorFlow
import CStbImage

func loadImage(url: URL) -> Tensor<Float> {
    var (width, height, channels): (Int32, Int32, Int32) = (0, 0, 0)
    guard let data = load_image(url.path, &width, &height, &channels, 0) else {
        fatalError("Failed to load image: \(url.path)")
    }
    defer { free_image(data) }
    
    let bp = UnsafeBufferPointer<UInt8>(start: data, count: Int(width*height*channels))
    let image = Tensor(bp.map { Float($0) / 255 })
        .reshaped(to: [Int(height), Int(width), Int(channels)])
    
    return image
}
