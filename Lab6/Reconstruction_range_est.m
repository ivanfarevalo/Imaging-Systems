close all;
quadrature_data = load('/Users/ivanarevalomac/Documents/MATLAB/Range Estimation/gpr_data.mat');
B = 1.024e9;
N = 128;
v = 3e8/sqrt(6);
n = linspace(1, 128, 128);
R = n.*v/(2*B)';


subim = zeros(40, 200);

freq = linspace(quadrature_data.f(1), quadrature_data.f(end), 1012);

vid = VideoWriter('newfile.avi');
open(vid)

for i = 1:size(quadrature_data.F,2)
    fft_func = (fft(conj(quadrature_data.F(:,i)), 1012));
    
    for y = 1:size(subim,1)
        for x = 1:size(subim,2)
            r = sqrt((i-x)^2 + y^2);
%             r = (r * 0.0213)+1;

            pxl_val = 0.5 * (fft_func(floor(r)) + fft_func(ceil(r)));
            
            f0 = 0.5 * (freq(floor(r)) + freq(ceil(r)) );
            
            pxl_val = pxl_val * exp(-1i*2*pi*f0*(2*r*0.0213)/v);
            subim(y,x) = subim(y,x) + pxl_val;
            

        end
    end
    
    frame = abs(subim) / max(abs(subim),[],'all');
    writeVideo(vid,frame);
    
%     if mod(i,25) == 0
%         image = imagesc(abs(subim));
%         title('Range profile');
%         xlabel('Position');
%         ylabel('Range');
%         pause(2)
%     end
    
end

close(vid)

image = imagesc([0,4], [0,0.2],abs(subim));
title('Range profile');
xlabel('Position');
ylabel('Range');
pause(2)
