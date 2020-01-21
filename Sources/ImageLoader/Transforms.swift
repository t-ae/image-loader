import TensorFlow

/// `Transform` takes rank 3 tensor which has shape [Height, Width, Channels] and mutate it.
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
    /// Smaller edge will be resized to `size`.
    @available(*, deprecated, message: "Use Transforms.resize")
    public static func resizeBilinear(aspectFill size: Int) -> Transform {
        return { image in
            let (height, width) = (image.shape[0], image.shape[1])
            let (newHeight, newWidth): (Int, Int)
            if height > width {
                newWidth = size
                newHeight = size * height / width
            } else {
                newHeight = size
                newWidth = size * width / height
            }
            image = _Raw.resizeBilinear(
                images: image.expandingShape(at: 0),
                size: Tensor<Int32>([Int32(newHeight), Int32(newWidth)])
            ).squeezingShape(at: 0)
        }
    }
    
    /// Resize image with keeping aspect ratio.
    /// Larger edge will be resized to `size`.
    @available(*, deprecated, message: "Use Transforms.resize")
    public static func resizeBilinear(aspectFit size: Int) -> Transform {
        return { image in
            let (height, width) = (image.shape[0], image.shape[1])
            let (newHeight, newWidth): (Int, Int)
            if height < width {
                newWidth = size
                newHeight = size * height / width
            } else {
                newHeight = size
                newWidth = size * width / height
            }
            image = _Raw.resizeBilinear(
                images: image.expandingShape(at: 0),
                size: Tensor<Int32>([Int32(newHeight), Int32(newWidth)])
            ).squeezingShape(at: 0)
        }
    }
    
    /// Resize image.
    @available(*, deprecated, message: "Use Transforms.resize")
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
            let w = image.shape[1]
            let h = image.shape[0]
            
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
            let w = image.shape[1]
            let h = image.shape[0]
            
            let y0 = (height - h) / 2
            let y1 = (height - h) - y0
            let x0 = (width - w) / 2
            let x1 = (width - w) - x0
            
            image = image.padded(forSizes: [(y0, y1), (x0, x1), (0, 0)], with: paddingValue)
        }
    }
    
    /// Flip vertically at random.
    ///
    /// - Note: Currently this doesn't support reproducibility.
    public static func randomFlipVertically() -> Transform {
        return { image in
            if Bool.random() {
                image = _Raw.reverseV2(image, axis: Tensor<Int32>([0]))
            }
        }
    }
    
    /// Flip horizontally at random.
    ///
    /// - Note: Currently this doesn't support reproducibility.
    public static func randomFlipHorizontally() -> Transform {
        return { image in
            if Bool.random() {
                image = _Raw.reverseV2(image, axis: Tensor<Int32>([1]))
            }
        }
    }
    
    /// Resize image with keeping aspect ratio.
    ///
    /// Smaller edge will be resized to `size`.
    public static func resize(_ method: ResizeMethod, aspectFill size: Int) -> Transform {
        return { image in
            let (height, width) = (image.shape[0], image.shape[1])
            let (newHeight, newWidth): (Int, Int)
            if height > width {
                newWidth = size
                newHeight = size * height / width
            } else {
                newHeight = size
                newWidth = size * width / height
            }
            image = method.resize(image: image, width: newWidth, height: newHeight)
        }
    }
    
    /// Resize image with keeping aspect ratio.
    ///
    /// Larger edge will be resized to `size`.
    public static func resize(_ method: ResizeMethod, aspectFit size: Int) -> Transform {
        return { image in
            let (height, width) = (image.shape[0], image.shape[1])
            let (newHeight, newWidth): (Int, Int)
            if height < width {
                newWidth = size
                newHeight = size * height / width
            } else {
                newHeight = size
                newWidth = size * width / height
            }
            image = method.resize(image: image, width: newWidth, height: newHeight)
        }
    }
    
    public static func resize(_ method: ResizeMethod, width: Int, height: Int) -> Transform {
        return { image in
            image = method.resize(image: image, width: width, height: height)
        }
    }
}

public enum ResizeMethod {
    case nearestNeighbor, bilinear, bicubic, area
    
    func resize(image: Tensor<Float>, width: Int, height: Int) -> Tensor<Float> {
        var image = image.expandingShape(at: 0)
        let size = Tensor<Int32>([Int32(height), Int32(width)])
        switch self {
        case .nearestNeighbor:
            image = _Raw.resizeNearestNeighbor(images: image, size: size)
        case .bilinear:
            image = _Raw.resizeBilinear(images: image, size: size)
        case .bicubic:
            image = _Raw.resizeBicubic(images: image, size: size)
        case .area:
            image = _Raw.resizeArea(images: image, size: size)
        }
        return image.squeezingShape(at: 0)
    }
}
