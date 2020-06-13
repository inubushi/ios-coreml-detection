//
//  ViewController.swift
//  CoreML Detection
//
//  Created by Chamin Morikawa on 2020/05/25.
//  Copyright © 2020 Chamin Morikawa. All rights reserved.
//

import UIKit
import CoreML
import Vision

class ViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    
    @IBOutlet weak var imgViewPhoto: UIImageView!
    @IBOutlet weak var labelInfo: UILabel!
    @IBOutlet weak var activityIndicator: UIActivityIndicatorView!
    
    // using the YOLO V3 model, because a model file was provided by Apple
    let model = YOLOv3FP16()
    
    // class labels are not contained in the model :o(. I had to enter them myself
    // Not that I invented them, these are available on Github, mostly inside Python code
    var classLabels: [String] = ["Person", "Bicycle", "Car", "Motorbike", "Aeroplane", "Bus", "Train", "Truck", "Boat",
                                 "Traffic Light", "Fire Hydrant", "Stop Sign", "Parking Meter", "Bench", "Bird", "Cat",
                                 "Dog", "Horse", "Sheep", "Cow", "Elephant", "Bear", "Zebra", "Giraffe", "Backpack",
                                 "Umbrella", "Handbag", "Tie", "Suitcase", "Frisbee", "Skis", "Snowboard", "Sports Ball",
                                 "Kite", "Baseball Bat", "Baseball Glove", "Skateboard”, “Surfboard”, “Tennis Racket",
                                 "Bottle", "Wine Glass", "Cup", "Fork", "Knife", "Spoon", "Bowl", "Banana", "Apple",
                                 "Sandwich”, “Orange”, “Broccoli”, “Carrot”, “Hot Dog", "Pizza", "Donut", "Cake",
                                 "Chair", "Sofa", "Potted Plant", "Bed", "Dining Table", "Toilet", "TV Monitor", "Laptop",
                                 "Mouse","Remote", "Keyboard", "Cellphone", "Microwave", "Oven", "Toaster", "Sink",
                                 "Refrigerator", "Book", "Clock", "Vase", "Scissors", "Teddy Bear", "Hair Drier", "Toothbrush"]
    
    // CALayer for showing results
    var resultsLayer: CALayer! = nil
    private var detectionOverlay: CALayer! = nil
    
    var selectedPhoto:UIImage!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
    }
    
    override func viewWillAppear(_ animated: Bool) {
        // if a sample has been selected, classify it
        if (selectedPhoto != nil) {
            // set image
            imgViewPhoto.image = selectedPhoto
            // start classification in background
            labelInfo.text = "Analyzing Image..."
            activityIndicator.startAnimating()
            
            DispatchQueue.global(qos: .background).async {
                self.detectAndDraw(img: self.selectedPhoto)
            }
        }
    }

    override func viewDidLayoutSubviews() {
        // setup layers for visualization
        setupLayers()
    }
    //MARK: Button Events
    // use camera to take a photo for classification
    @IBAction func cameraButtonTapped(_ sender: Any) {
        // cannot take photos on simulator
        if !UIImagePickerController.isSourceTypeAvailable(.camera) {
            return
        }
        
        // clear previous sample, if any
        selectedPhoto = nil
        
        // open camera
        let cameraPicker = UIImagePickerController()
        cameraPicker.delegate = self
        cameraPicker.sourceType = .camera
        cameraPicker.allowsEditing = true
        
        present(cameraPicker, animated: true)
    }
    
    //MARK: Image Picker Delegate
    // show the selected photo and perform classification
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        
        picker.dismiss(animated: true)
        
        // set image
        guard let image = info[UIImagePickerController.InfoKey.editedImage] as? UIImage else {
            return
        }
        imgViewPhoto.image = image
        
        // start classification in background
        labelInfo.text = "Analyzing Image..."
        activityIndicator.startAnimating()
        
        DispatchQueue.global(qos: .background).async {
            self.detectAndDraw(img: image)
        }
    }
    
    func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        // did not pick a photo
        dismiss(animated: true, completion: nil)
    }
    
    //MARK: Detection
    func detectAndDraw(img:UIImage)  {
        // conversion to the correct input size,
        // and get the image data to a pixel buffer
        UIGraphicsBeginImageContextWithOptions(CGSize(width: 416, height: 416), true, 2.0)
        img.draw(in: CGRect(x: 0, y: 0, width: 416, height: 416))
        let newImage = UIGraphicsGetImageFromCurrentImageContext()!
        UIGraphicsEndImageContext()
        
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue, kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue] as CFDictionary
        var pixelBuffer : CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault, Int(newImage.size.width), Int(newImage.size.height), kCVPixelFormatType_32ARGB, attrs, &pixelBuffer)
        guard (status == kCVReturnSuccess) else {
            return
        }
        
        CVPixelBufferLockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
        let pixelData = CVPixelBufferGetBaseAddress(pixelBuffer!)
        
        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
        let context = CGContext(data: pixelData, width: Int(newImage.size.width), height: Int(newImage.size.height), bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer!), space: rgbColorSpace, bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)
        
        context?.translateBy(x: 0, y: newImage.size.height)
        context?.scaleBy(x: 1.0, y: -1.0)
        
        UIGraphicsPushContext(context!)
        newImage.draw(in: CGRect(x: 0, y: 0, width: newImage.size.width, height: newImage.size.height))
        UIGraphicsPopContext()
        CVPixelBufferUnlockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
        
        // forward pass with time measurement
        let start = DispatchTime.now()
        
        // for now
        guard let prediction = try? model.prediction(image: pixelBuffer!, iouThreshold: 0.5, confidenceThreshold: 0.5) else {
            return
        }
        
        let end = DispatchTime.now()
        
        let nanoTime = end.uptimeNanoseconds - start.uptimeNanoseconds // processing time in nano seconds (UInt64)
        let timeInterval = Double(nanoTime) / 1_000_000 // convert to milliseconds, for ease of reading
        
        // update results on main thread
        DispatchQueue.main.async {
            // UI updates
            self.activityIndicator.stopAnimating()
            if prediction.confidence.count > 0 {
                // status message
                let resultString = String(format:"Detected %d objects in %f milliseconds", (prediction.confidence.count/self.classLabels.count), timeInterval)
                self.labelInfo.text = resultString
                // draw boxes around detected objects
                self.drawResults(prediction: prediction)
            } else {
                self.labelInfo.text = "AI could not detect anything that it has been trained to"
                CATransaction.begin()
                CATransaction.setValue(kCFBooleanTrue, forKey: kCATransactionDisableActions)
                self.detectionOverlay.sublayers = nil // remove all the old recognized objects
                CATransaction.commit()
            }
        }
    }
    
    // MARK: Visualization
    func setupLayers() {
        resultsLayer = imgViewPhoto.layer
        detectionOverlay = CALayer() // container layer that has all the renderings of the observations
        detectionOverlay.name = "DetectionOverlay"
        detectionOverlay.bounds = CGRect(x: 0.0,
                                         y: 0.0,
                                         width: imgViewPhoto.frame.size.width,
                                         height: imgViewPhoto.frame.size.height)
        detectionOverlay.position = CGPoint(x: resultsLayer.bounds.midX, y: resultsLayer.bounds.midY)
        resultsLayer.addSublayer(detectionOverlay)
    }
    
    func drawResults (prediction:YOLOv3FP16Output) {
        // if we are here, we do have things to draw.
        let boundingBoxes = prediction.coordinates
        let confidences = prediction.confidence
        
        CATransaction.begin()
        CATransaction.setValue(kCFBooleanTrue, forKey: kCATransactionDisableActions)
        
        // remove results of the previous detection, if any
        imgViewPhoto.layer.sublayers?.forEach { $0.removeFromSuperlayer() }
        setupLayers()
        
        // loop over each object
        let count = boundingBoxes.shape[0].intValue
        for i in 0 ... count-1 {
            let objectID:NSNumber = NSNumber.init(value: i)
            // get bounding box: there is more than one way to specify a bounding box
            // for this network, the center coordinates and the dimensions are used
            // they have been divided by the image height and width, so that the values are in the range [0,0, 1.0]
            let centerY = boundingBoxes[[objectID,0]]
            let centerX = boundingBoxes[[objectID,1]]
            let height = boundingBoxes[[objectID,2]]
            let width = boundingBoxes[[objectID,3]]
            
            // get confidences for all classes
            var confidenceVals:[Float] = []
            for j in 0...79 {
                confidenceVals.append(confidences[[objectID,NSNumber.init(value: j)]].floatValue)
            }
            
            // pick the class index with maximum confidence
            let maxConfidence: Float = confidenceVals.max()!
            let classIdx: Int = confidenceVals.firstIndex(of: maxConfidence)!
            
            // find the corresponding class label
            let classLabel:String = classLabels[classIdx]
            
            // create a CGRect to be drawn on the photo
            // also make sure that the dimensions don't go outside the frame dimensions
            var boxLeft = (CGFloat(truncating: centerX) - CGFloat(truncating: width)/2.0)*imgViewPhoto.frame.size.width
            if boxLeft < 0 {
                boxLeft = 0
            }
            var boxTop = (CGFloat(truncating: centerY) - CGFloat(truncating: height)/2.0)*imgViewPhoto.frame.size.height
            if boxTop < 0 {
                boxTop = 0
            }
            var boxWidth = CGFloat(truncating: width)*imgViewPhoto.frame.size.width
            if boxLeft + boxWidth >= imgViewPhoto.frame.size.width {
                boxWidth = imgViewPhoto.frame.size.width - 1.0 - boxLeft
            }
            var boxHeight = CGFloat(truncating: height)*imgViewPhoto.frame.size.height
            if boxLeft + boxWidth >= imgViewPhoto.frame.size.width {
                boxHeight = imgViewPhoto.frame.size.height - 1.0 - boxTop
            }
            
            let objectBounds: CGRect = CGRect.init(x: boxLeft, y: boxTop, width: boxWidth , height: boxHeight)
            
            // draw box
            let shapeLayer = self.createRoundedRectLayerWithBounds(objectBounds)
            
            // write label
            let textLayer = self.createTextSubLayerInBounds(objectBounds,
                                                            identifier: classLabel,
                                                            confidence: maxConfidence)
            
            // add them to the overlay
            shapeLayer.addSublayer(textLayer)
            detectionOverlay.addSublayer(shapeLayer)
        }
        
        self.updateLayerGeometry()
        CATransaction.commit()
        
    }
    
    func createTextSubLayerInBounds(_ bounds: CGRect, identifier: String, confidence: VNConfidence) -> CATextLayer {
        let textLayer = CATextLayer()
        textLayer.name = "Object Label"
        let formattedString = NSMutableAttributedString(string: String(format: "\(identifier)\nConfidence:  %.2f", confidence))
        let largeFont = UIFont(name: "Helvetica", size: 24.0)!
        formattedString.addAttributes([NSAttributedString.Key.font: largeFont], range: NSRange(location: 0, length: identifier.count))
        textLayer.string = formattedString
        textLayer.bounds = CGRect(x: 0, y: 0, width: bounds.size.height - 10, height: bounds.size.width - 10)
        textLayer.position = CGPoint(x: bounds.midX, y: bounds.midY)
        textLayer.shadowOpacity = 0.7
        textLayer.shadowOffset = CGSize(width: 2, height: 2)
        textLayer.foregroundColor = CGColor(colorSpace: CGColorSpaceCreateDeviceRGB(), components: [0.0, 0.0, 0.0, 1.0])
        textLayer.contentsScale = 2.0 // retina rendering
        // rotate the layer into screen orientation and scale and mirror
        textLayer.setAffineTransform(CGAffineTransform(rotationAngle: CGFloat(.pi / 2.0)).scaledBy(x: 1.0, y: -1.0))
        return textLayer
    }
    
    func createRoundedRectLayerWithBounds(_ bounds: CGRect) -> CALayer {
        let shapeLayer = CALayer()
        shapeLayer.bounds = bounds
        shapeLayer.position = CGPoint(x: bounds.midX, y: bounds.midY)
        shapeLayer.name = "Found Object"
        shapeLayer.backgroundColor = CGColor(colorSpace: CGColorSpaceCreateDeviceRGB(), components: [1.0, 1.0, 0.2, 0.4])
        shapeLayer.cornerRadius = 7
        return shapeLayer
    }
    
    func updateLayerGeometry() {
        let bounds = resultsLayer.bounds
        var scale: CGFloat
        
        let xScale: CGFloat = bounds.size.width / imgViewPhoto.frame.size.height
        let yScale: CGFloat = bounds.size.height / imgViewPhoto.frame.size.width
        
        scale = fmax(xScale, yScale)
        if scale.isInfinite {
            scale = 1.0
        }
        CATransaction.begin()
        CATransaction.setValue(kCFBooleanTrue, forKey: kCATransactionDisableActions)
        
        // rotate the layer into screen orientation and scale and mirror
        detectionOverlay.setAffineTransform(CGAffineTransform(rotationAngle: CGFloat(.pi / 2.0)).scaledBy(x: scale, y: -scale))
        // center the layer
        detectionOverlay.position = CGPoint(x: bounds.midX, y: bounds.midY)
        
        CATransaction.commit()
        
    }
    
    //MARK: Select Sample
    func setSelectedPhoto(img:UIImage) {
        selectedPhoto = img
    }

}

