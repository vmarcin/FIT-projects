function [h,p,r] = hist2(y1,x) 

L = length(x); 
N = length(y1); 

h = zeros(L,L);

xcol = x(:); bigx = repmat(xcol,1,N); 

yr = y1(:)'; bigy = repmat(yr,L,1);
[~,ind1] = min(abs(bigy - bigx)); 
ind2 = ind1(11:N);

for ii=1:N-10,
  d1 = ind1(ii);   d2 = ind2(ii); 
  h(d1,d2) = h(d1,d2) + 1; 
end

surf = (x(2) - x(1))^2; 
p = h / N / surf;  

x = x(:); X1 = repmat(x,1,L);

x = x'; X2 = repmat(x,L,1); 

r = sum(sum (X1 .* X2 .* p)) * surf;

check = sum(sum (p)) * surf; 
disp(['hist2: check -- 2d integral should be 1 and is ' num2str(check)]); 