%priklad 1
[s,Fs] = audioread('xmarci10.wav');s=s';
fprintf('vzorkovacia frekvencia je %d Hz\n',Fs);
N=length(s); %dlzka vo vzorkoch
fprintf('dlzka signalu vo vzorkoch: %d\n',N);
fprintf('dlzka signalu v sekundach: %d\n',N/Fs); %dlzka v sekundach

%priklad 2
G = 10 * log10(1/N * abs(fft(s)).^2);
f = (0:N/2-1)/N * Fs; G = G(1:N/2);
plot (f,G); xlabel('f[Hz]'); ylabel('PSD[dB]'); grid;

%priklad 3
%https://ch.mathworks.com/help/matlab/ref/max.html
[M,I] = max(G);
I = I - 1; %index maxima (matlab indexuje od 1)
fprintf('maximum modulu spektra je %g a nachadza sa na frekvencii %d Hz\n',M,I);

%priklad 4
b = [0.2324 -0.4112 0.2324]; a = [1 0.2289 0.4662];
zplane(b,a);
p = roots(a);
if (isempty(p) | abs(p) < 1)
    disp('STABILNY :)')
else
    disp('NESTABILNY :(')
end

%priklad 5
H = freqz(b,a,Fs/2);
plot (f,abs(H)); xlabel('f[Hz]'); ylabel('|H(f)|'); grid;

%priklad 6
h = filter(b,a,s);
G = 10 * log10(1/N * abs(fft(h)).^2); G = G(1:N/2);
plot(f,G); xlabel('f[Hz]'); ylabel('PSD[dB]'); grid;

%priklad 7
[M,I] = max(G);
I = I - 1; %index maxima (matlab indexuje od 1)
fprintf('maximum modulu spektra je %g a nachadza sa na frekvencii %d\n',M,I);

%priklad 8
%https://ch.mathworks.com/matlabcentral/answers/215704-xcorr-how-to-find-the-location-of-the-highest-correlation-on-the-actual-data-using-xcorr
%https://ch.mathworks.com/help/signal/ref/xcorr.html
a = [1 1 -1 -1];
a = repmat(a,1,80);%20ms obdlznikovych signalov
[c,lag] = xcorr(s,a);
[M,I] = max(abs(c));
idx = lag(I) + 1;
time =idx/Fs;
fprintf('obdlznikove impulzy sa nachadzaju od casu %d vo vzorkach a %g v sekundach\n',idx,time);

%priklad 9
Rv = xcorr(s,'biased'); k = -50:50;
plot(k,Rv(N-50:N+50));xlabel('k'); ylabel('R[k]');grid;

%priklad 10
R10 = Rv(N+10);
fprintf('R[10]=%g\n',R10);

%priklady 11. 12. 13.
gmin = min(s);
gmax = max(s);
x = linspace(gmin,gmax,50);
[h,p,r] = hist2(s,x);
bar3(p);
axis([0 51 0 51 0 3]); xlabel('x1'); ylabel('x2');
fprintf('R[10]=%g\n',r);




