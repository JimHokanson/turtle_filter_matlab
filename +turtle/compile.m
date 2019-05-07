c = mex.compilers.gcc('filter.c','verbose',true);
%c.addCompileFlags({'-std=c11','-mavx','-mfma','-ffast-math','-O3'});
c.addCompileFlags({'-std=c11','-ffast-math','-O3'});


%from big_plot.compile - is this for openmp?
if strcmp(c.gcc_type,'mingw64')
    c.addStaticLibs({'pthread'})
end

c.addLib('openmp');
c.build();