import TensorFlow

public typealias Transform = (inout Tensor<Float>)->Void

public enum Transforms {
    /// Crop specified size image from center.
    public static func centerCrop(width: Int, height: Int) -> Transform {
        return { image in
            let (h, w, c) = (image.shape[0], image.shape[1], image.shape[2])
            precondition(height <= h, "`height` is larger than image height.")
            precondition(width <= w, "`width` is larger than image width.")
            
            let y = (h - height) / 2
            let x = (w - width) / 2
            image = image.slice(lowerBounds: [y, x, 0], upperBounds: [y+height, x+width, c])
        }
    }
    
    /// Resize image with keeping aspect ratio.
    /// Smaller edge will be `smallerSize`.
    public static func resizeBilinear(aspectFill smallerSize: Int) -> Transform {
        return { image in
            let (height, width) = (image.shape[0], image.shape[1])
            let (newHeight, newWidth): (Int, Int)
            if height > width {
                newWidth = smallerSize
                newHeight = smallerSize * height / width
            } else {
                newHeight = smallerSize
                newWidth = smallerSize * width / height
            }
            image = _Raw.resizeBilinear(
                images: image.expandingShape(at: 0),
                size: Tensor<Int32>([Int32(newHeight), Int32(newWidth)])
            ).squeezingShape(at: 0)
        }
    }
    
    /// Resize image.
    public static func resizeBilinear(width: Int, height: Int) -> Transform {
        return { image in
            image = _Raw.resizeBilinear(
                images: image.expandingShape(at: 0),
                size: Tensor<Int32>([Int32(height), Int32(width)])
            ).squeezingShape(at: 0)
        }
    }
    
    /// Add padding to make image square
    public static func paddingToSquare(with paddingValue: Float) -> Transform {
        return { image in
            let w = image.shape[0]
            let h = image.shape[1]
            
            if w == h {
                // Nothing to do
            } else if w > h {
                let y0 = (w-h) / 2
                let y1 = (w-h) - y0
                image = image.padded(forSizes: [(y0, y1), (0, 0), (0, 0)], with: paddingValue)
            } else {
                let x0 = (h-w) / 2
                let x1 = (h-w) - x0
                image = image.padded(forSizes: [(0, 0), (x0, x1), (0, 0)], with: paddingValue)
            }
        }
    }
    
    /// Add padding to make image specified size
    public static func paddingTo(width: Int, height: Int, with paddingValue: Float) -> Transform {
        return { image in
            let w = image.shape[0]
            let h = image.shape[1]
            
            let y0 = (height - h) / 2
            let y1 = (height - h) - y0
            let x0 = (width - w) / 2
            let x1 = (width - w) - x0
            
            image = image.padded(forSizes: [(y0, y1), (x0, x1), (0, 0)], with: paddingValue)
        }
    }
}
