
import gin

@gin.configurable
class TrainerConfig( object ) :

    def __init__( self,
                  expertPolicyFile = 'experts/Ant-v2.pkl',
                  expertDataFile = 'data/experts/ant_500.pkl',
                  environmentName = 'Ant-v2',
                  learningRate = 0.005,
                  batchSize = 64,
                  numEpochs = 10,
                  seed = 0,
                  sessionID = 'session_ant',
                  backend = 'pytorch' ) :
        super( TrainerConfig, self ).__init__()

        self.expertPolicyFile = expertPolicyFile
        self.expertDataFile = expertDataFile
        self.environmentName = environmentName
        self.learningRate = learningRate
        self.batchSize = batchSize
        self.numEpochs = numEpochs
        self.seed = seed
        self.sessionID = sessionID
        self.backend = backend


@gin.configurable
class BackboneConfig( object ) :

    def __init__( self,
                  observationsShape = (4,),
                  actionsShape = (4,),
                  layersDefs = [] ) :
        super( BackboneConfig, self ).__init__()

        self.observationsShape = observationsShape
        self.actionsShape = actionsShape
        self.layersDefs = layersDefs