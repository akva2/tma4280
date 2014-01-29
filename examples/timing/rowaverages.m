function avg = rowaverages(C)
  avg = zeros(size(C,1),1);
  for i=1:size(C,1)
    avg(i) = sum(C(i,:))/size(C,2);
  end
