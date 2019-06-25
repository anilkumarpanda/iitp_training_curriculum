def compute_slope(*args):
    """
    Computes the slope of a linear regression
     
    
    Arguments:
       * args (list of tuples) containing the x,y coordinates
        
    Returns:
        c (float): slope
    """
   
  
    xs = [el[0] for el in args]
    ys = [el[1] for el in args]
    
    xs_squared =  [el[0]*el[0] for el in args]
    xys =  [el[0]*el[1] for el in args]
    n = len(args)
    
    
    num = n*sum(xys) - sum(xs)*sum(ys)
    denom = n*sum(xs_squared)- sum(xs)*sum(xs)

    return num/denom

def compute_intercept(*args):
    """
    Computes the intercept of a linear regression
     
    
    Arguments:
       * args (list of tuples) containing the x,y coordinates
        
    Returns:
        c (float): slope
    """
    xs = [el[0] for el in args]
    ys = [el[1] for el in args]
    
    xs_squared =  [el[0]*el[0] for el in args]
    xys =  [el[0]*el[1] for el in args]
    n = len(args)
    

    num = sum(ys) * sum(xs_squared) - sum(xs)*sum(xys)
    denom = n*sum(xs_squared) - sum(xs)*sum(xs)
      
    return num/denom