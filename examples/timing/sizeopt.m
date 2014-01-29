function sizeopt(typename, tablename, sizefile, blasfile)
  sizes=load(sizefile);
  avg = zeros(4, length(sizes));
  for i=1:4
    s = sprintf('%s_%i.asc',typename,i-1);
    data = load(s);
    avg(i,:) = rowaverages(data)';
  end
  data = load(blasfile);
  avg_blas = rowaverages(data);

  f = fopen(tablename,'w');
  fprintf(f,'\\begin{center}\n');
  fprintf(f,'\t\\begin{tabular}{|c|c|c|c|c||c|}\n');
  fprintf(f,'\t\t\\hline\n');
  fprintf(f,'\t\t n & -O0 & -O1 & -O2 & -O3 & BLAS\\\\\n');
  fprintf(f,'\t\t\\hline\n');
  for i=1:size(avg,2)
    fprintf(f, '\t\t %i & %f & %f & %f & %f & %f', sizes(i), avg(1,i), avg(2,i), avg(3,i), avg(4,i), avg_blas(i));
    fprintf(f,'\\\\\n');
  end
  fprintf(f,'\t\t\\hline\n');
  fprintf(f,'\t\\end{tabular}\n');
  fprintf(f,'\\end{center}\n');
  fclose(f);
