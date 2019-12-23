import TensorFlow

public enum Transforms {
    /// Crop specified size image from center.
    public static func centerCrop(width: Int, height: Int) -> ImageLoader.Transform {
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
    public static func resizeBilinear(aspectFill smallerSize: Int) -> ImageLoader.Transform {
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
    public static func resizeBilinear(width: Int, height: Int) -> ImageLoader.Transform {
        return { image in
            image = _Raw.resizeBilinear(
                images: image.expandingShape(at: 0),
                size: Tensor<Int32>([Int32(height), Int32(width)])
            ).squeezingShape(at: 0)
        }
    }
}
