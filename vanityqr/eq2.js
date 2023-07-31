function f(x, y, t){
  let theta = Math.atan2(y, x);
  if ( theta < 0 ) {
    theta += Math.PI * 2
  }
  let theta45 = theta
  while ( theta45 >= Math.PI / 2.0 ) {
    theta45 -= Math.PI / 2.0
  }
  const theta_side = Math.tan( (theta45 >= Math.PI / 4 )? Math.PI / 2 - theta45: theta45)

  const L = 1//Math.sqrt( 1 + theta_side * theta_side )
  const D = Math.min( Math.sqrt( x*x + y*y ), 1)
  const T = (Math.cos( D / L * Math.PI) + 1) / 2.0

  const x0 = 0
  const x1 = 0.3
  const y0 = 0.4
  const y1 = 0.8
  const V = 1

  let aa = 0
  let bb = 0
  let tt = 0
  const pi2 = Math.PI / 2
  if ( theta <= pi2 ) {
    aa = x1
    bb = y1
    //tt = theta / pi2
    tt = (Math.cos( Math.PI - theta * 2 ) + 1) / 2
  }
  else if ( theta <= 2 * pi2 ) {
    aa = y1
    bb = x0
    //tt = (theta - pi2) / pi2
    tt = (Math.cos( Math.PI - (theta - pi2) * 2 ) + 1) / 2
  }
  else if ( theta <= 3 * pi2 ) {
    aa = x0
    bb = y0
    //tt = (theta - pi2 * 2) / pi2
    tt = (Math.cos( Math.PI - (theta - pi2 * 2) * 2 ) + 1) / 2
  }
  else if ( theta <= 4 * Math.PI ) {
    aa = y0
    bb = x1
    //tt = (theta - pi2 * 3) / pi2
    tt = (Math.cos( Math.PI - (theta - pi2 * 3) * 2 ) + 1) / 2
  }
  const E = aa * (1 - tt) + bb * tt
  return V * T + E * (1 - T)
}
