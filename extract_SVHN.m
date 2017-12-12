clc; clear;

load train_32x32.mat;

folder = 'SVHN_train';

mkdir(folder);
for i = 0:9
   mkdir(strcat(folder, '/', num2str(i)));
end

num_im = length(y);

counts = zeros(1, 10);

for i = 1:num_im
   im = X(:,:,:,i);
   label = y(i);
   if label == 10
       label = 0;
   end
   
   imwrite(im,strcat(folder, '/', num2str(label), '/', sprintf('%07d', counts(label+1)), '.png'));
   
   counts(label+1) = counts(label+1) + 1;
   
   if mod(i, 10000) == 0
      disp(i) 
   end
   
end