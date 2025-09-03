#!/usr/bin/env python3
"""
Image Comparison Script for FRUGAL Integration Validation
Compare inference.jpg (neural network output) with reference.jpg (input sampling)
"""

import sys
import os

def compare_images(img1_path, img2_path):
    """Compare two images and return similarity metrics."""
    print(f"🔍 Comparing images:")
    print(f"   Image 1: {img1_path}")
    print(f"   Image 2: {img2_path}")
    print()
    
    # Check if files exist
    if not os.path.exists(img1_path):
        print(f"❌ ERROR: Image 1 not found: {img1_path}")
        return False
        
    if not os.path.exists(img2_path):
        print(f"❌ ERROR: Image 2 not found: {img2_path}")
        return False
    
    # Get file sizes
    size1 = os.path.getsize(img1_path)
    size2 = os.path.getsize(img2_path)
    print(f"📊 File sizes:")
    print(f"   {os.path.basename(img1_path)}: {size1:,} bytes")
    print(f"   {os.path.basename(img2_path)}: {size2:,} bytes")
    print()
    
    try:
        # Try importing required libraries
        try:
            import cv2
            import numpy as np
            opencv_available = True
        except ImportError:
            print("⚠️  OpenCV not available, trying PIL...")
            opencv_available = False
            try:
                from PIL import Image
                import numpy as np
                pil_available = True
            except ImportError:
                print("❌ Neither OpenCV nor PIL available for image comparison")
                print("   Install with: pip install opencv-python  OR  pip install pillow")
                return None
        
        # Load images
        if opencv_available:
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)
            
            if img1 is None:
                print(f"❌ Cannot read image with OpenCV: {img1_path}")
                return False
                
            if img2 is None:
                print(f"❌ Cannot read image with OpenCV: {img2_path}")
                return False
                
            print(f"✅ Images loaded with OpenCV")
            
        else:  # Use PIL
            img1_pil = Image.open(img1_path)
            img2_pil = Image.open(img2_path)
            img1 = np.array(img1_pil)
            img2 = np.array(img2_pil)
            
            print(f"✅ Images loaded with PIL")
        
        # Check dimensions
        print(f"📐 Image dimensions:")
        print(f"   {os.path.basename(img1_path)}: {img1.shape}")
        print(f"   {os.path.basename(img2_path)}: {img2.shape}")
        
        if img1.shape != img2.shape:
            print(f"⚠️  Image dimensions differ - comparison may not be accurate")
            # Resize to same dimensions if possible
            if opencv_available:
                h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
                img1 = cv2.resize(img1, (w, h))
                img2 = cv2.resize(img2, (w, h))
                print(f"   Resized both to: {img1.shape}")
        
        print()
        
        # Calculate basic metrics
        print("🔢 Quantitative Comparison:")
        
        # Mean Squared Error
        mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
        print(f"   MSE: {mse:.2f}")
        
        if mse > 0:
            # Peak Signal-to-Noise Ratio
            if opencv_available:
                try:
                    psnr = cv2.PSNR(img1, img2)
                    print(f"   PSNR: {psnr:.2f} dB")
                except:
                    max_pixel = 255.0
                    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
                    print(f"   PSNR: {psnr:.2f} dB (calculated)")
            else:
                max_pixel = 255.0
                psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
                print(f"   PSNR: {psnr:.2f} dB")
        else:
            psnr = float('inf')
            print(f"   PSNR: ∞ (identical images)")
        
        # Try SSIM if scikit-image is available
        try:
            from skimage.metrics import structural_similarity as ssim
            
            # Convert to grayscale for SSIM if color images
            if len(img1.shape) == 3:
                if opencv_available:
                    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                else:
                    # Convert RGB to grayscale
                    gray1 = np.dot(img1[...,:3], [0.2989, 0.5870, 0.1140])
                    gray2 = np.dot(img2[...,:3], [0.2989, 0.5870, 0.1140])
                
                ssim_score = ssim(gray1.astype(np.uint8), gray2.astype(np.uint8))
            else:
                ssim_score = ssim(img1, img2)
                
            print(f"   SSIM: {ssim_score:.4f}")
            
        except ImportError:
            print("   SSIM: Not available (install scikit-image for structural similarity)")
            ssim_score = None
        
        print()
        
        # Pixel difference analysis
        diff = np.abs(img1.astype(float) - img2.astype(float))
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        print("🎨 Pixel Analysis:")
        print(f"   Max pixel difference: {max_diff:.1f}")
        print(f"   Mean pixel difference: {mean_diff:.1f}")
        
        # Count significantly different pixels
        significant_diff = np.sum(diff > 50)  # Pixels differing by more than 50
        total_pixels = diff.size
        diff_percentage = (significant_diff / total_pixels) * 100
        
        print(f"   Significantly different pixels: {significant_diff:,} ({diff_percentage:.1f}%)")
        print()
        
        # Interpretation
        print("🎯 Interpretation:")
        
        if mse < 100:
            print("   ✅ Images are very similar (MSE < 100)")
            status = "VERY_SIMILAR"
        elif mse < 1000:
            print("   ✅ Images are reasonably similar (MSE < 1000)")
            status = "SIMILAR"
        elif mse < 5000:
            print("   ⚠️  Images have moderate differences (MSE < 5000)")
            status = "MODERATE_DIFF"
        else:
            print("   ❌ Images are significantly different (MSE ≥ 5000)")
            status = "VERY_DIFFERENT"
        
        if ssim_score is not None:
            if ssim_score > 0.9:
                print("   ✅ Excellent structural similarity (SSIM > 0.9)")
            elif ssim_score > 0.7:
                print("   ✅ Good structural similarity (SSIM > 0.7)")
            elif ssim_score > 0.5:
                print("   ⚠️  Moderate structural similarity (SSIM > 0.5)")
            else:
                print("   ❌ Poor structural similarity (SSIM ≤ 0.5)")
        
        if diff_percentage < 10:
            print("   ✅ Most pixels are similar (< 10% significantly different)")
        elif diff_percentage < 25:
            print("   ⚠️  Some pixels differ significantly (< 25%)")
        else:
            print("   ❌ Many pixels differ significantly (≥ 25%)")
        
        print()
        
        # Overall assessment for FRUGAL validation
        print("🏆 FRUGAL Validation Assessment:")
        
        if status in ["VERY_SIMILAR", "SIMILAR"]:
            print("   ✅ VALIDATION CONCERN: Images are too similar!")
            print("   📝 Expected: inference.jpg should be DIFFERENT from reference.jpg")
            print("   📝 reference.jpg = input sampling, inference.jpg = learned output")
            print("   📝 If they're very similar, the neural network may not have trained properly")
            return "TOO_SIMILAR"
            
        elif status == "MODERATE_DIFF":
            print("   ✅ GOOD: Images show moderate differences")
            print("   📝 This suggests the neural network learned and generated a different output")
            print("   📝 FRUGAL optimization appears to be working correctly")
            return True
            
        else:  # VERY_DIFFERENT
            print("   ⚠️  CAUTION: Images are very different")
            print("   📝 This could indicate:")
            print("   📝   1. Normal - neural network learned dramatically different representation")
            print("   📝   2. Problem - FRUGAL corrupted the computation")
            print("   📝 Manual visual inspection recommended")
            return "VERY_DIFFERENT"
        
    except Exception as e:
        print(f"❌ Error during image comparison: {e}")
        return False

def main():
    """Main function to run image comparison."""
    
    print("🖼️  FRUGAL Integration Image Comparison")
    print("=" * 50)
    print()
    
    if len(sys.argv) != 3:
        print("Usage: python3 compare_images.py <image1> <image2>")
        print()
        print("Example:")
        print("  python3 compare_images.py reference.jpg inference.jpg")
        print()
        print("For FRUGAL validation:")
        print("  reference.jpg  = sampled input image")
        print("  inference.jpg  = neural network output (FRUGAL optimized)")
        print()
        sys.exit(1)
    
    img1_path = sys.argv[1]
    img2_path = sys.argv[2]
    
    result = compare_images(img1_path, img2_path)
    
    print("🎉 Comparison Complete!")
    print("=" * 50)
    
    if result is True:
        print("✅ FRUGAL VALIDATION: PASSED")
        print("   Neural network appears to have learned properly with FRUGAL optimization")
        sys.exit(0)
    elif result == "TOO_SIMILAR":
        print("⚠️  FRUGAL VALIDATION: POTENTIAL ISSUE")
        print("   Images are suspiciously similar - check if neural network trained properly")
        sys.exit(2)
    elif result == "VERY_DIFFERENT":
        print("⚠️  FRUGAL VALIDATION: NEEDS REVIEW") 
        print("   Images are very different - manual inspection recommended")
        sys.exit(3)
    elif result is False:
        print("❌ FRUGAL VALIDATION: FAILED")
        print("   Technical error during comparison")
        sys.exit(1)
    else:
        print("⚠️  FRUGAL VALIDATION: INCONCLUSIVE")
        print("   Unable to perform detailed comparison - install required libraries")
        sys.exit(4)

if __name__ == "__main__":
    main()