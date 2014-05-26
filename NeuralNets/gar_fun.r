################################################################################
# This function calculates relative predictor importance weights for a neural
# network.
#
# Original code is available here:
#   https://gist.github.com/fawda123/6206737
#
# A couple of slight modifications/bug fixes have been put in place to work with
# our data.
#
################################################################################

gar.fun<-function(out.var,mod.in,bar.plot=T,x.names=NULL,...){
  
  #gets weights for neural network, output is list
  #if rescaled argument is true, weights are returned but rescaled based on abs value
  nnet.vals<-function(mod.in,nid,rel.rsc,struct.out=struct){
    
    require(scales)
    require(reshape)
    
    if('numeric' %in% class(mod.in)){
      struct.out<-struct
      wts<-mod.in
    }
    
    #neuralnet package
    if('nn' %in% class(mod.in)){
      struct.out<-unlist(lapply(mod.in$weights[[1]],ncol))
      struct.out<-struct.out[-length(struct.out)]
      struct.out<-c(
        length(mod.in$model.list$variables),
        struct.out,
        length(mod.in$model.list$response)
      )    		
      wts<-unlist(mod.in$weights[[1]])   
    }
    
    #nnet package
    if('nnet' %in% class(mod.in)){
      struct.out<-mod.in$n
      wts<-mod.in$wts
    }
    
    #RSNNS package
    if('mlp' %in% class(mod.in)){
      struct.out<-c(mod.in$nInputs,mod.in$archParams$size,mod.in$nOutputs)
      hid.num<-length(struct.out)-2
      wts<-mod.in$snnsObject$getCompleteWeightMatrix()
      
      #get all input-hidden and hidden-hidden wts
      inps<-wts[grep('Input',row.names(wts)),grep('Hidden_2',colnames(wts)),drop=F]
      inps<-melt(rbind(rep(NA,ncol(inps)),inps))$value
      uni.hids<-paste0('Hidden_',1+seq(1,hid.num))
      for(i in 1:length(uni.hids)){
        if(is.na(uni.hids[i+1])) break
        tmp<-wts[grep(uni.hids[i],rownames(wts)),grep(uni.hids[i+1],colnames(wts)),drop=F]
        inps<-c(inps,melt(rbind(rep(NA,ncol(tmp)),tmp))$value)
      }
      
      #get connections from last hidden to output layers
      outs<-wts[grep(paste0('Hidden_',hid.num+1),row.names(wts)),grep('Output',colnames(wts)),drop=F]
      outs<-rbind(rep(NA,ncol(outs)),outs)
      
      #weight vector for all
      wts<-c(inps,melt(outs)$value)
      assign('bias',F,envir=environment(nnet.vals))
    }
    
    if(nid) wts<-rescale(abs(wts),c(1,rel.rsc))
    
    #convert wts to list with appropriate names 
    hid.struct<-struct.out[-c(length(struct.out))]
    row.nms<-NULL
    for(i in 1:length(hid.struct)){
      if(is.na(hid.struct[i+1])) break
      row.nms<-c(row.nms,rep(paste('hidden',i,seq(1:hid.struct[i+1])),each=1+hid.struct[i]))
    }
    row.nms<-c(
      row.nms,
      rep(paste('out',seq(1:struct.out[length(struct.out)])),each=1+struct.out[length(struct.out)-1])
    )
    out.ls<-data.frame(wts,row.nms)
    out.ls$row.nms<-factor(row.nms,levels=unique(row.nms),labels=unique(row.nms))
    out.ls<-split(out.ls$wts,f=out.ls$row.nms)
    
    assign('struct',struct.out,envir=environment(nnet.vals))
    
    out.ls
    
  }
  
  best.wts<-nnet.vals(mod.in,nid=F,rel.rsc=5,struct.out=NULL)
  out.ind<-which(out.var==colnames(eval(mod.in$call$y)))
  
  #get input-hidden weights and hidden-output weights, remove bias
  inp.hid<-data.frame(
    do.call('cbind',best.wts[grep('hidden',names(best.wts))])[-1,],
    row.names=c(colnames(eval(mod.in$call$x)))
  )
  idx <- which(paste('out', out.ind) == names(best.wts))
  hid.out<-best.wts[[idx]][-1]
#   hid.out<-best.wts[[grep(paste('out',out.ind),names(best.wts))]][-1]
  
  #multiply hidden-output connection for each input-hidden weight
  mult.dat<-data.frame(
    sapply(1:ncol(inp.hid),function(x) inp.hid[,x]*hid.out[x]),
    row.names=rownames(inp.hid)
  )    
  names(mult.dat)<-colnames(inp.hid)
  
  #get relative contribution of each input variable to each hidden node, sum values for each input
  #inp.cont<-rowSums(apply(mult.dat,2,function(x) abs(x)/sum(abs(x))))
  inp.cont<-rowSums(mult.dat)
  
  #get relative contribution
  #inp.cont/sum(inp.cont)
  
  rel.imp<-{
    signs<-sign(inp.cont)
    signs*rescale(abs(inp.cont),c(0,1))
  }
  
  if(is.null(x.names)) x.names<-colnames(eval(mod.in$call$x))
  
  if(bar.plot){
    mp<-barplot(sort(rel.imp),names='',...)
    text(mp,-1.15,srt=45,adj=1,labels=x.names[order(rel.imp)],xpd=T)
  }
  
  list(
    mult.wts=mult.dat,
    inp.cont=inp.cont,
    rel.imp=rel.imp
  )
  
}