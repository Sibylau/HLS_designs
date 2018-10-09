function A = genA(n)
% Generate a dense n x n symmetric, positive definite matrix

A = rand(n, n); % generate a random n x n matrix
A = A+A'; 
A = A + n*eye(n);

str = strcat('op',num2str(n),'.bin');
fp=fopen(str,'w');
fwrite(fp,A,'float');
fclose(fp);



end
