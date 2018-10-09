function A = genA(m, n)
%generate m-by-n full-rank random matrix A
A = randi(100, m, n);

str = strcat('op', num2str(m), 'x', num2str(n), '.bin');
fp = fopen(str,'wb');
fwrite(fp, A, 'float');
fclose(fp);

end
