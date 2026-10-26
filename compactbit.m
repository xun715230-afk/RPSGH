function cb = compactbit(b)
% 将二进制矩阵转换为紧凑位表示
[nSamples, nBits] = size(b);
nWords = ceil(nBits/8);
cb = zeros([nSamples nWords], 'uint8');

for i = 1:nBits
    w = ceil(i/8);
    cb(:,w) = bitset(cb(:,w), mod(i-1,8)+1, b(:,i));
end
end