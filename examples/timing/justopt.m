function justopt(typename, tablename)
  avg = zeros(4, 1);
  for i=1:4
    s = sprintf('%s_%i.asc',typename,i-1);
    data = load(s);
    avg(i,:) = rowaverages(data)';
  end
  f = fopen(tablename,'w');
  fprintf(f,'\\begin{center}\n');
  fprintf(f,'\t\\begin{tabular}{|c|c|c|c|c|}\n');
  fprintf(f,'\t\t\\hline\n');
  fprintf(f,'\t\t n & -O0 & -O1 & -O2 & -O3\\\\\n');
  fprintf(f,'\t\t\\hline\n');
  fprintf(f, '\t\t %i & %f & %f & %f & %f\\\\\n\t\t\\hline\n', 10, avg(1), avg(2), avg(3), avg(4));
  fprintf(f,'\t\\end{tabular}\n');
  fprintf(f,'\\end{center}');
  fclose(f);
