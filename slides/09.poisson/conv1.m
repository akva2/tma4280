function conv1
  pts = linspace(10, 1000, 10);
  ers = zeros(size(pts));
  for i=1:length(pts)
    ers(i) = poisson1D(pts(i), 0, 1, @source1, @exact1);
  end
  figure; loglog(pts, ers, 'bx-', 'linewidth', 2); hold on; 
  reference = 10.^([-2 -4 -6]);
  loglog([10^1 10^2 10^3],reference,'ko-','linewidth',2);
  legend('convergence rate','second order reference');
  set(gca, 'fontsize',18);
