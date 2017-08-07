% Creation of the function that extracts SIFT features from the images
function [f,d]=computeSIFTfeatures(I)
I=single(rgb2gray(I));
I=single(mat2gray(I));
[f,d]=vl_sift(I); 
f=transpose(f);
[m,n]=size(f);
d=ones(m,1);
end