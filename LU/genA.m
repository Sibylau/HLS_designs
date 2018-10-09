function A = genA(n)
%generate n-by-n full-rank random matrix A
A = randi(100, n, n);

str = strcat('op', num2str(n), '.bin');
fp = fopen(str,'wb');
fwrite(fp, A, 'float');
fclose(fp);

end
