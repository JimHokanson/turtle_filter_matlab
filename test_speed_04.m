function test_speed_04()
%

MIN_ORDER = 7;
MAX_ORDER = 10;
N = 5;
data_length = 1e7;
x = rand(1,data_length);
x = ones(1,data_length);
x = zeros(1,data_length);
x(2:3:end) = 1;
x(3:3:end) = -1;
times = zeros(N,4,MAX_ORDER);

for i = 1:N
    for j = MIN_ORDER:MAX_ORDER
        %[b,a] = cheby1(j,3,2000/100000);
        [b,a] = cheby2(j,3,2000/100000);
        
        fprintf('Iteration %d for order %d\n',i,j);
        tic
        y1 = filter(b,a,x);
        times(i,1,j) = toc;
        
        tic
        y1 = sl.array.mex_filter(b,a,x);
        times(i,2,j) = toc;
        
        tic
        y2 = turtle.filter(b,a,x,0);
        times(i,3,j) = toc;
        
        tic
        y3 = turtle.filter(b,a,x,2);
        times(i,4,j) = toc;
    end
end

t1 = squeeze(median(times(:,1,:)));
t2 = squeeze(median(times(:,2,:)));
t3 = squeeze(median(times(:,3,:)));
t4 = squeeze(median(times(:,4,:)));

figure(4)
clf
subplot(1,2,1)
plot(1000*t1)
hold on
plot(1000*t2)
plot(1000*t3)
plot(1000*t4)
hold off
legend({'ML','JS','Me0','Me1'})
ylabel(sprintf('Execution times (ms)'))
title(sprintf('IIR for data length %d',data_length))


subplot(1,2,2)
plot(t1./t3)
hold on
plot(t2./t3)
hold off
ylabel('Speedup')
legend({'vs ML','vs JS'})

keyboard


end