image_path='../ShanghaiTech/part_B/test_data/images';
position_head_path='../ShanghaiTech/part_B/test_data/ground-truth';
pmap_path='../ShanghaiTech/part_B/test_data/Pmap';
for i=1:316
    imgPath=fullfile(image_path,num2str(i,'IMG_%d.jpg'));
    posPath=fullfile(position_head_path,num2str(i,'GT_IMG_%d.mat'));
    pmapPath=fullfile(pmap_path,num2str(i,'PMAP_%d.mat'));
    pmap=segmentation(imgPath, posPath);
    save(pmapPath,'pmap');
    imagesc(pmap);
    %set(gca,'XTick',[]);
    %set(gca,'YTick',[]);
    %set(gca,'Position',[0 0 1 1]);
    %saveas(gcf,['./PMAP_',num2str(i),'.jpg'])
end