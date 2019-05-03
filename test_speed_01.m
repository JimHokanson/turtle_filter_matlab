function test_speed_01()
%
%   Notes: memcpy on mac takes 100 ms for 1e8


%FIR speeds:  Me  ML   JS
%Std, openMP: 480 1693 2311
%             369 1079 1975
%             369 1033 1953
%             

N = 5;
x = rand(1,1e8);
[b,a] = cheby1(7,3,2000/100000);

times = zeros(N,3);
for i = 1:N
    tic
    y1 = filter(b,1,x);
    times(i,1) = toc;
    
    tic
    y2 = sl.array.mex_filter(b,1,x);
    times(i,2) = toc;
    
    tic
    y3 = turtle.filter(b,1,x);
    times(i,3) = toc;
end

% 
% %How much slower when 'a' is used?
% %about 2x as long
% tic
% for i = 1:N
%     y = filter(b,a,x);
% end
% t4 = toc/N;
% 
% tic
% for i = 1:N
%     y = sl.array.mex_filter(b,a,x);
% end
% t5 = toc/N;

t1 = median(times(:,1));
t2 = median(times(:,2));
t3 = median(times(:,3));

fprintf('%04.1f : ML FIR\n',1000*t1);
fprintf('%04.1f : JS FIR\n',1000*t2);
fprintf('%04.1f : Me FIR\n',1000*t3);
% fprintf('%04.1f : ML IIR\n',1000*t4);
% fprintf('%04.1f : JS IIR\n',1000*t5);


keyboard

plotBig(y1-y3)


end