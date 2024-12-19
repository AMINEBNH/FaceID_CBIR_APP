from descriptor import glcm
from distances import euclidean, manhattan, chebyshev, canberra

path = 'images/test.png'
patha = 'images/testa.png'
pathb = 'images/testb.png'

def main():
    feat_glcm_k = glcm(path)
    feat_glcm_a = glcm(patha)
    feat_glcm_b = glcm(pathb)
    
    print(f'''Euclidean:\nd(path, patha)={euclidean(feat_glcm_k, feat_glcm_a)}\nd(path, pathb) 
          = {euclidean(feat_glcm_k, feat_glcm_b)}''')
    print(f'''Manhattan:\nd(path, patha)={manhattan(feat_glcm_k, feat_glcm_a)}\nd(path, pathb) 
          = {manhattan(feat_glcm_k, feat_glcm_b)}''')
    print(f'''Chebyshev:\nd(path, patha)={chebyshev(feat_glcm_k, feat_glcm_a)}\nd(path, pathb) 
          = {chebyshev(feat_glcm_k, feat_glcm_b)}''')
    print(f'''Canberra:\nd(path, patha)={canberra(feat_glcm_k, feat_glcm_a)}\nd(path, pathb) 
          = {canberra(feat_glcm_k, feat_glcm_b)}''')

if __name__ == '__main__':
    main()