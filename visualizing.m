
%% Visualizing the eigen images
figure;
for j = 1:24
    im = V(:,j);
    im = reshape(im,n_row,n_col);    
    im = (im - min(im(:)))/(max(im(:)) - min(im(:)));
    subplot(4,6,j);
    imshow(im);
end

%% Visualizing reconstructed  version of images
figure;
for j = 1:25
    im = X_recon(:,j);
    im = reshape(im,n_row,n_col);    
    im = (im - min(im(:)))/(max(im(:)) - min(im(:)));
    subplot(5,5,j);
    imshow(im);
end

%% Simple visualizing the dataset images
figure;
title('Example Images from IITB data-set');
for j = 7:9
    im_dest = strcat('0',string(j),'.jpg');
    im = imread(im_dest);       
    subplot(1,4,j-6);
    imshow(im);
end





