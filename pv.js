// ============================================================
// 0. 全局参数配置
// ============================================================
var GLOBAL_SEED = 42;
var WORK_SCALE = 30;

// 掩膜与阈值
var RURAL_THRESHOLD = 5;        
var RURAL_FOCAL_RADIUS = 3;
var CONF_THRESHOLD = 0.7;       

// 采样配置
var POINTS_PER_TRUE_POLY = 20; 
var TRUE_POINTS_CAP = 1200;     
var SPLIT_RATIO = 0.7;          // 70% 训练，30% 验证

// --- 矩形采样参数 ---
var POS_RECT_SIZE = 60;   // 正样本矩形边长 60m
var NEG_RECT_SIZE = 90;   // 负样本矩形边长 90m

// 后处理配置
var MIN_PATCH_PIXELS = 4;       // 面积筛选阈值

// ============================================================
// 1. 数据准备
// ============================================================
var roi = table.geometry(); 
Map.centerObject(roi, 11);

// 1.1 加载特征与 APV_Index
var featureStackRaw = ee.Image("users/nathanchen011126/APV_fullFeature_db_s2");
var apvIndexImg = ee.Image("users/nathanchen011126/APV_Index_db").select('APV_Index');

var featureStack = featureStackRaw.addBands(apvIndexImg);
var bands = featureStack.bandNames();
var baseProj = featureStackRaw.projection();

// 1.2 制作乡村掩膜
var viirs = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
  .filterDate('2023-01-01', '2023-12-31')
  .select('avg_rad')
  .median();

var rural_mask = viirs.lt(RURAL_THRESHOLD)
  .reproject({crs: baseProj, scale: 500})
  .focal_max({radius: RURAL_FOCAL_RADIUS, units: 'pixels'})
  .clip(roi);

// 1.3 应用掩膜
var baseImage = featureStack.updateMask(rural_mask).clip(roi).select(bands);

var embeddingRaw = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
  .filterDate('2023-01-01', '2024-01-01')
  .filterBounds(roi)
  .median();

var embeddingImage = embeddingRaw
  .resample('bilinear')
  .reproject({crs: baseProj, scale: WORK_SCALE})
  .updateMask(rural_mask)
  .clip(roi);

var embeddingBands = embeddingImage.bandNames();
var fullFeatureImage = baseImage.addBands(embeddingImage);
var requiredBands = bands.cat(embeddingBands);

// ============================================================
// 2. 样本划分 (Train/Val Split) 与 矩形采样
// ============================================================
var truePolygons = ee.FeatureCollection("users/nathanchen011126/real_agripv")
  .filterBounds(roi);
var falsePointsSource = ee.FeatureCollection("users/nathanchen011126/fake_nega_db_balance")
  .filterBounds(roi);

var truePolysWithRand = truePolygons.randomColumn('split_rand', GLOBAL_SEED);
var trainPolys = truePolysWithRand.filter(ee.Filter.lt('split_rand', SPLIT_RATIO));
var valPolys = truePolysWithRand.filter(ee.Filter.gte('split_rand', SPLIT_RATIO));

var falseSourceWithRand = falsePointsSource.randomColumn('split_rand', GLOBAL_SEED);
var trainFalseSource = falseSourceWithRand.filter(ee.Filter.lt('split_rand', SPLIT_RATIO));
var valFalseSource = falseSourceWithRand.filter(ee.Filter.gte('split_rand', SPLIT_RATIO));

// --- 辅助函数：将点转为矩形 ---
var pointToRect = function(feature, size, classLabel) {
  var geom = feature.geometry();
  // buffer 半径为 size/2，然后取 bounds 得到外接矩形
  var rect = geom.buffer(size / 2).bounds(); 
  return ee.Feature(rect).set('class', classLabel);
};

// --- 生成训练集 (矩形) ---
// 1. 正样本：先在多边形内撒点，再把点变矩形
var trainTrueN = ee.Number(trainPolys.size()).multiply(POINTS_PER_TRUE_POLY).min(TRUE_POINTS_CAP).int();
var trainTruePtsRaw = ee.FeatureCollection.randomPoints({
  region: trainPolys.geometry(), points: trainTrueN, seed: GLOBAL_SEED
});
// 映射为矩形
var trainTrueRects = trainTruePtsRaw.map(function(f) {
  return pointToRect(f, POS_RECT_SIZE, 1);
});

// 2. 负样本：直接取点，再把点变矩形
var trainFalseN = trainTrueN.multiply(3);
var trainFalsePtsRaw = trainFalseSource.limit(trainFalseN);
// 映射为矩形
var trainFalseRects = trainFalsePtsRaw.map(function(f) {
  return pointToRect(f, NEG_RECT_SIZE, 0);
});

var trainPolysFinal = trainTrueRects.merge(trainFalseRects);

// --- 生成验证集 (矩形) ---
// 1. 正样本
var valTrueN = ee.Number(valPolys.size()).multiply(POINTS_PER_TRUE_POLY).int(); 
var valTruePtsRaw = ee.FeatureCollection.randomPoints({
  region: valPolys.geometry(), points: valTrueN, seed: GLOBAL_SEED + 1
});
var valTrueRects = valTruePtsRaw.map(function(f) {
  return pointToRect(f, POS_RECT_SIZE, 1);
});

// 2. 负样本
var valFalseN = valTrueN.multiply(3);
var valFalsePtsRaw = valFalseSource.limit(valFalseN);
var valFalseRects = valFalsePtsRaw.map(function(f) {
  return pointToRect(f, NEG_RECT_SIZE, 0);
});

var valPolysFinal = valTrueRects.merge(valFalseRects);

// --- 提取特征 (注意：tileScale 调大以防止矩形采样内存溢出) ---
var trainingDataAll = fullFeatureImage.sampleRegions({
  collection: trainPolysFinal, 
  properties: ['class'], 
  scale: WORK_SCALE, 
  tileScale: 16, 
  geometries: true // 保留几何信息以便调试（可选）
});

var trainingDataClean = trainingDataAll.filter(ee.Filter.notNull(requiredBands)).filter(ee.Filter.notNull(['class']));

print('训练集样本数(像素级):', trainingDataClean.size());

// ============================================================
// 3. 模型训练
// ============================================================
var classifierBase = ee.Classifier.smileRandomForest(400).train({
  features: trainingDataClean.select(bands.add('class')),
  classProperty: 'class', inputProperties: bands
});

var classifierEmbed = ee.Classifier.smileRandomForest(200).train({
  features: trainingDataClean.select(embeddingBands.add('class')),
  classProperty: 'class', inputProperties: embeddingBands
});

// ============================================================
// 4. 分类与后处理 (Morphology -> Area -> Vector)
// ============================================================
var probBase = fullFeatureImage.classify(classifierBase.setOutputMode('PROBABILITY'));
var probEmbed = fullFeatureImage.classify(classifierEmbed.setOutputMode('PROBABILITY'));

var rawClass = probBase.gte(CONF_THRESHOLD).and(probEmbed.gte(CONF_THRESHOLD)).rename('raw_class');

// --- 步骤 1: 形态学滤波 (Morphological Filtering) ---
var postMode = rawClass.focal_mode({radius: 1.5, kernelType: 'square', units: 'pixels'});

// --- 步骤 2: 面积筛选 (Area Filtering) ---
var patchSize = postMode.selfMask().connectedPixelCount(10);
var finalClassified = postMode
  .updateMask(patchSize.gte(MIN_PATCH_PIXELS)) // 只保留大于阈值的图斑
  .unmask(0)
  .rename('final_class')
  .updateMask(rural_mask)
  .clip(roi)
  .uint8();

// --- 步骤 3: 转为矢量 (Convert to Vector) ---
// 仅提取值为 1 (光伏) 的区域转为矢量，以减小数据量
var vectors = finalClassified
  .updateMask(finalClassified.eq(1)) // 关键：只保留光伏像素，忽略背景
  .reduceToVectors({
    geometry: roi,
    crs: finalClassified.projection(),
    scale: WORK_SCALE,
    geometryType: 'polygon',
    eightConnected: false, // false 通常能得到更整洁的边界
    labelProperty: 'class',
    reducer: ee.Reducer.countEvery(),
    maxPixels: 1e13,
    bestEffort: true
  });

print('生成的矢量图斑数量:', vectors.size());

// ============================================================
// 5. 精度验证 (Standard Validation)
// ============================================================
print('--- 开始标准验证评估 ---');
// 验证集同样使用矩形采样
var valDataAll = finalClassified.sampleRegions({
  collection: valPolysFinal, 
  properties: ['class'], 
  scale: WORK_SCALE, 
  tileScale: 16
});
var valClean = valDataAll.filter(ee.Filter.notNull(['class', 'final_class']));

var cm = valClean.errorMatrix('class', 'final_class');
var cmArray = cm.array();
var TN = cmArray.get([0, 0]), FP = cmArray.get([0, 1]);
var FN = cmArray.get([1, 0]), TP = cmArray.get([1, 1]);

var eps = 1e-9;
var OA = ee.Number(TP).add(TN).divide(ee.Number(TP).add(TN).add(FP).add(FN).add(eps));
var Precision = ee.Number(TP).divide(ee.Number(TP).add(FP).add(eps));
var Recall = ee.Number(TP).divide(ee.Number(TP).add(FN).add(eps));
var F1 = ee.Number(2).multiply(Precision).multiply(Recall).divide(Precision.add(Recall).add(eps));

print('验证集混淆矩阵:', cm);
print('OA:', OA, 'F1:', F1);

// ============================================================
// 6. 导出结果 (矢量 + 报告)
// ============================================================

// 导出矢量结果 (Shapefile)
Export.table.toDrive({
  collection: vectors,
  description: 'APV_Result_Vector_Shapefile',
  folder: 'GEE_Results',
  fileFormat: 'SHP'
});

// 导出标准精度报告
var accuracyFeature = ee.Feature(null, {
  'Scenario': 'Train 70% / Val 30% (Rect Sample)',
  'Ratio_Neg_Pos': '3:1',
  'TN': TN, 'FP': FP, 'FN': FN, 'TP': TP,
  'OA': OA, 'Precision': Precision, 'Recall': Recall, 'F1': F1
});

Export.table.toDrive({
  collection: ee.FeatureCollection([accuracyFeature]),
  description: 'APV_Accuracy_Report',
  folder: 'GEE_Reports',
  fileFormat: 'CSV'
});

// ============================================================
// 7. 新增：空间留一法 (SLOO) / 空间 K 折交叉验证模块
// ============================================================
print('--- 开始空间留一法(SLOO)验证 ---');

// --- 7.1 参数配置 ---
var SLOO_FOLDS = 5;           // 折数 (K-Fold)，建议 5-10
var SLOO_BUFFER = 200;       // 空间缓冲区距离 (米)，用于隔离训练和验证集
var SLOO_SEED = 123;          // 独立随机种子

// --- 7.2 准备数据 (合并所有样本并重新分配 Fold) ---
// 将训练集和验证集的矩形合并，用于整体评估
var allPolys = trainPolysFinal.merge(valPolysFinal);

// 为每个矩形分配一个随机的 Fold ID (0 到 SLOO_FOLDS-1)
var allPolysWithFold = allPolys.randomColumn('fold_rand', SLOO_SEED)
  .map(function(f) {
    var fold = ee.Number(f.get('fold_rand')).multiply(SLOO_FOLDS).floor().toInt();
    return f.set('fold', fold);
  });

// --- 7.3 定义交叉验证过程 ---
// 注意：这里去掉了末尾的 .flatten()，先保留为 List
var slooResultsList = ee.List.sequence(0, SLOO_FOLDS - 1).map(function(k) {
  var foldId = ee.Number(k);
  
  // 1. 定义验证集 (Test Fold)
  var testPolys = allPolysWithFold.filter(ee.Filter.eq('fold', foldId));
  
  // 2. 定义潜在训练集 (Train Folds)
  var potentialTrainPolys = allPolysWithFold.filter(ee.Filter.neq('fold', foldId));
  
  // 3. 空间隔离 (剔除缓冲区内的训练多边形)
  var spatialFilter = ee.Filter.withinDistance({
    distance: SLOO_BUFFER,
    leftField: '.geo',
    rightField: '.geo',
    maxError: 10
  });
  
  var invertedJoin = ee.Join.inverted();
  var trainPolysSpatiallyIndependent = invertedJoin.apply(potentialTrainPolys, testPolys, spatialFilter);
  
  // 4. 提取像素
  var trainPixels = fullFeatureImage.sampleRegions({
    collection: trainPolysSpatiallyIndependent,
    properties: ['class'],
    scale: WORK_SCALE,
    tileScale: 16,
    geometries: false
  }).filter(ee.Filter.notNull(requiredBands));
  
  var testPixels = fullFeatureImage.sampleRegions({
    collection: testPolys,
    properties: ['class'],
    scale: WORK_SCALE,
    tileScale: 16,
    geometries: false
  }).filter(ee.Filter.notNull(requiredBands));

  // 5. 训练临时模型 (使用较少的树以加快 CV 速度)
  var rfBaseSLOO = ee.Classifier.smileRandomForest(50).train({
    features: trainPixels.select(bands.add('class')),
    classProperty: 'class',
    inputProperties: bands
  });
  
  var rfEmbedSLOO = ee.Classifier.smileRandomForest(30).train({
    features: trainPixels.select(embeddingBands.add('class')),
    classProperty: 'class',
    inputProperties: embeddingBands
  });
  
  // 6. 批量预测
  var predictedBase = testPixels.classify(rfBaseSLOO.setOutputMode('PROBABILITY'), 'probs_base');
  var predictedEmbed = predictedBase.classify(rfEmbedSLOO.setOutputMode('PROBABILITY'), 'probs_embed');
  
  // 7. 解析概率并应用阈值 (修复了 Float/List 类型报错问题)
  return predictedEmbed.map(function(feat) {
    // 获取属性值
    var valBase = feat.get('probs_base');
    var valEmbed = feat.get('probs_embed');

    // --- 关键修复：鲁棒的概率获取函数 ---
    // 逻辑：如果是 List，取 index 1 (Class 1 的概率)；如果是 Number，直接使用
    var getProb = function(val) {
      var isList = ee.Algorithms.IsEqual(ee.Algorithms.ObjectType(val), 'List');
      return ee.Number(ee.Algorithms.If(
        isList,
        ee.List(val).get(1), // 如果是 List
        val                  // 如果是 Float
      ));
    };

    var pBase = getProb(valBase);
    var pEmbed = getProb(valEmbed);
    
    // 双重阈值判断
    var isAgri = pBase.gte(CONF_THRESHOLD).and(pEmbed.gte(CONF_THRESHOLD)).toInt();
    
    return feat.set('sloo_pred', isAgri);
  });
});

// 关键修正：将 List 变为 FeatureCollection 后再 flatten()
// 这样才能把 5 个 Collection 里的像素合并成一个大的 Collection
var slooResultCol = ee.FeatureCollection(slooResultsList).flatten();

// 调试输出：检查一下是否有数据
print('SLOO 总验证样本数:', slooResultCol.size());

// --- 7.4 计算并输出 SLOO 混淆矩阵 ---
// 强制指定 order: [0, 1] 避免 1x1 矩阵报错
var slooCM = slooResultCol.errorMatrix('class', 'sloo_pred', [0, 1]); 

var slooAcc = slooCM.accuracy();
var slooKappa = slooCM.kappa();

// 获取矩阵数值
var matrixArray = slooCM.array(); 
var TN_val = matrixArray.get([0, 0]); 
var FP_val = matrixArray.get([0, 1]); 
var FN_val = matrixArray.get([1, 0]); 
var TP_val = matrixArray.get([1, 1]); 

print('============================================');
print('SLOO (Spatial K-Fold) 验证结果 (' + SLOO_FOLDS + '折, Buffer=' + SLOO_BUFFER + 'm):');
print('SLOO 混淆矩阵:', slooCM);
print('SLOO 总体精度 (OA):', slooAcc);
print('SLOO Kappa:', slooKappa);
print('TP:', TP_val, 'TN:', TN_val, 'FP:', FP_val, 'FN:', FN_val);
print('============================================');

// 导出 SLOO 精度报告
var slooAccuracyFeature = ee.Feature(null, {
  'Scenario': 'SLOO Validation (K=' + SLOO_FOLDS + ', Buf=' + SLOO_BUFFER + 'm)',
  'OA': slooAcc,
  'Kappa': slooKappa,
  'TP': TP_val,
  'TN': TN_val,
  'FP': FP_val,
  'FN': FN_val
});

Export.table.toDrive({
  collection: ee.FeatureCollection([slooAccuracyFeature]),
  description: 'APV_SLOO_Accuracy_Report',
  folder: 'GEE_Reports',
  fileFormat: 'CSV'
});