# app.py
from descriptor import glcm

def main():
    path = 'images/test.png'
    
    # GLCM Features
    try:
        feat_glcm = glcm(path)
        print("GLCM Features:")
        print(f"Dissimilarity: {feat_glcm[0]}")
        print(f"Contrast: {feat_glcm[1]}")
        print(f"Correlation: {feat_glcm[2]}")
        print(f"Energy: {feat_glcm[3]}")
        print(f"ASM: {feat_glcm[4]}")
        print(f"Homogeneity: {feat_glcm[5]}")
    except FileNotFoundError as e:
        print(e)
    except ValueError as e:
        print(e)
    
    print()
    
    # # Bitdesc Features
    # try:
    #     feat_bitdesc = bitdesc(path)
    #     print("Bitdesc Features:")
    #     for key, value in feat_bitdesc.items():
    #         print(f"{key.capitalize()}: {value}")
    # except FileNotFoundError as e:
    #     print(e)
    # except ValueError as e:
    #     print(e)

if __name__ == "__main__":
    main()
