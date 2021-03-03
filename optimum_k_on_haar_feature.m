close all; clear all
tic
training = load('training.mat'); training = training.out_training_matrix;
testing = load('testing.mat'); testing = testing.out_testing_matrix;
accuracy = zeros(0,0);
for K = 50:50:500
    N = size(training,3); n_row = size(training,1); n_col = size(training,2);
    X = zeros(n_row*n_col,0);
    %% Training Part
    for j = 1:N       
        im = training(:,:,j); 
        im_vec = im(:); X(:,end+1) = im_vec;    
    end %step1: vectorizing
    mean_mat = sum(X,2)/N; mean_mat = repmat(mean_mat,1,N); X = X - mean_mat; %step2: mean zero
    L = X'*X; %step3: L is the NxN matrix
    [V,D] = eig(L); [d,ind] = sort(diag(D),'descend');
    D = D(ind,ind); V = V(:,ind); %step4: obtaining ordered eigen vectors ans coeffs for L
    V = X*V; %step5: eigen vector of XX^T(i.e. dxd) 
    sq_sum = sqrt(sum(V.*V,1)); sq_sum = repmat(sq_sum,n_col*n_row,1); 
    V = V./sq_sum; %step6: unit normalizing the columns
    V_approx = V(:,1:K); 
    alpha = V_approx'*X; %gives all the required alpha_i for i = 1...N
    mean_vec = mean_mat(:,1); 
    %alpha, mean_vec and V_approx will be needed for testing
    X_recon = V_approx*alpha;

    %% testing Part
    X_test = zeros(n_row*n_col,0);
    N_test = size(testing,3);
    for j = 1:N_test       
        im = testing(:,:,j); 
        im_vec = im(:); X_test(:,end+1) = im_vec;    
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
xlabel('Accuracy of PCA'); ylabel('Number of Eigen Vectors Chosen');
title('Plot of Accuracy for PCA on Haar Features');
