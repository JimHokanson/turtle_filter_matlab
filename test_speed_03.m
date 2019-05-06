function test_speed_03()
%
%   Notes: memcpy on mac takes 100 ms for 1e8


%FIR speeds:  Me  ML   JS
%Std, openMP: 369, 1033, 1953 Mac
%Std, openMP: 296, 433, 1273
%             

ORDER = 4;
N = 10;
x = rand(1,1e8);
[b,a] = cheby1(ORDER,3,2000/100000);

times = zeros(N,3);
for i = 1:N
    tic
    y1 = filter(b,a,x);
    times(i,1) = toc;
    
    tic
    y2 = sl.array.mex_filter(b,a,x);
    times(i,2) = toc;
    
    tic
    y3 = turtle.filter(b,a,x);
    times(i,3) = toc;
end

t1 = median(times(:,1));
t2 = median(times(:,2));
t3 = median(times(:,3));

fprintf('%04.1f : ML FIR\n',1000*t1);
fprintf('%04.1f : JS FIR\n',1000*t2);
fprintf('%04.1f : Me FIR\n',1000*t3);


keyboard

plotBig(y1-y3)


end