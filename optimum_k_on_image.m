close all; clear all
tic
dec_level = 0;
accuracy = zeros(0,0);
n_row = 224/(2^dec_level); n_col = 224/(2^dec_level);
N = 600;
for K = 50:50:500
    X = zeros(n_row*n_col,0);
    %% Training Part
    for j = 1:100
        for k = 1:6
            if j<= 99 && j >= 10 
                im_dest = strcat('TrainAligned/0',string(j),'/0',string(k),'.png');                
            end
            if j <= 9
                im_dest = strcat('TrainAligned/00',string(j),'/0',string(k),'.png');        
            end
            if j == 100
                im_dest = strcat('TrainAligned/',string(j),'/0',string(k),'.png');                
            end        
            im = imread(im_dest); im = rgb2gray(im); 
            %[im,cH,cV,cD] = dwt2(im,'haar');
            im_vec = im(:); X(:,end+1) = im_vec; 
        end    
    end %step1: vectorizing
    mean_mat = sum(X,2)/N; mean_mat = repmat(mean_mat,1,N); X = X - mean_mat; %step2: mean zero
    L = X'*X; %step3: L is the NxN matrix
    [V,D] = eig(L); [d,ind] = sort(diag(D),'descend');
    D = D(ind,ind); V = V(:,ind); %step4: obtaining ordered eigen vectors ans coeffs for L
    V = X*V; %step5: eigen vector of XX^T(i.e. dxd) 
    sq_sum = sqrt(sum(V.*V,1)); sq_sum = repmat(sq_sum,n_col*n_row,1); 
    V = V./sq_sum; %step6: unit normalizing the columns
    V_approx = V(:,4:K); 
    alpha = V_approx'*X; %gives all the required alpha_i for i = 1...N
    mean_vec = mean_mat(:,1); 
    %alpha, mean_vec and V_approx will be needed for testing
    X_recon = V_approx*alpha;

    %% Testing Part
    X_test = zeros(n_row*n_col,0);
    N_test = 400;
    for j = 1:100
        for k = 7:10
            if k == 10
                if j <= 9
                    im_dest = strcat('TestAligned/00',string(j),'/',string(k),'.png');
                elseif j <= 99
                    im_dest = strcat('TestAligned/0',string(j),'/',string(k),'.png');
                else
                    im_dest = strcat('TestAligned/',string(j),'/',string(k),'.png');
                end
            else
                if j <= 9
                    im_dest = strcat('TestAligned/00',string(j),'/0',string(k),'.png');
                elseif j <= 99
                    im_dest = strcat('TestAligned/0',string(j),'/0',string(k),'.png');
                else
                    im_dest = strcat('TestAligned/',string(j),'/0',string(k),'.png');
                end            
            end        
            im = imread(im_dest); im = rgb2gray(im);
            %[im,cH,cV,cD] = dwt2(im,'haar');
            im_vec = im(:); X_test(:,end+1) = im_vec;         
        end    
    end
    X_test = X_test - repmat(mean_vec,1,N_test); %put mean to zero for test matrix 
    alpha_test = V_approx'*X_test;

    pred = zeros(N_test,2);
    for j = 1:N_test
        my_alpha = alpha_test(:,j);
        my_alpha = repmat(my_alpha,1,N);
        diff = alpha - my_alpha; 
        diff = diff.^2;
        %diff = abs(diff);
        diff = sum(diff,1);
        [val,ind] = min(diff);
        pred(j,1) = ceil(ind/6);
        pred(j,2) = ceil(j/4);
    end
    toc
    accuracy(end+1) = 100*(sum(pred(:,1) == pred(:,2))/N_test);
end

figure;
plot(50:50:500,accuracy,'LineWidth',3); hold on; grid on;
plot(50:50:500,accuracy,'o','MarkerSize',8);
xlabel('Number of Eigen Vectors Chosen'); ylabel('Accuracy of PCA');
title('Plot of Accuracy for PCA on Images');


