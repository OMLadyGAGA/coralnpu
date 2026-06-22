package bus

import chisel3._
import chisel3.util._

import coralnpu.Parameters


class TlulSocketM1(
    p: TLULParameters,
    M: Int = 4,
    HReqPass: Seq[Boolean] = Nil,
    HRspPass: Seq[Boolean] = Nil,
    HReqDepth: Seq[Int] = Nil,
    HRspDepth: Seq[Int] = Nil,
    DReqPass: Boolean = true,
    DRspPass: Boolean = true,
    DReqDepth: Int = 1,
    DRspDepth: Int = 1,
    moduleName: String = "TlulSocketM1"
) extends Module {
  override val desiredName = moduleName
  val StIdW = log2Ceil(M)

  // 1. Define device-side parameters (augmented source ID width)
  val p_d = {
    val mock_p = new coralnpu.Parameters()
    mock_p.lsuDataBits = p.w * 8
    mock_p.axi2IdBits = p.o + StIdW
    new bus.TLULParameters(mock_p)
  }

  val io = IO(new Bundle {
    val tl_h = Flipped(Vec(M, new OpenTitanTileLink.Host2Device(p)))
    val tl_d = new OpenTitanTileLink.Host2Device(p_d)
  })

  // 2. Combinational Request Path (A-Channel)
  // Augment source ID with host index combinationally before arbitration
  val hreq_a = Wire(Vec(M, Decoupled(new OpenTitanTileLink.A_Channel(p_d))))
  for (i <- 0 until M) {
    hreq_a(i).valid := io.tl_h(i).a.valid
    io.tl_h(i).a.ready := hreq_a(i).ready
    hreq_a(i).bits := io.tl_h(i).a.bits
    hreq_a(i).bits.source := Cat(io.tl_h(i).a.bits.source, i.U(StIdW.W))
  }

  // Arbiter selects one request combinationally
  val arb = Module(new Arbiter(new OpenTitanTileLink.A_Channel(p_d), M))
  for (i <- 0 until M) {
    arb.io.in(i) <> hreq_a(i)
  }

  // Drive device request directly (0 latency)
  io.tl_d.a <> arb.io.out

  // 3. Combinational Response Path (D-Channel)
  // Response steering based purely on the returning source ID
  val rsp_valid = io.tl_d.d.valid
  val rsp_bits = io.tl_d.d.bits
  val host_index = rsp_bits.source(StIdW - 1, 0)

  for (i <- 0 until M) {
    io.tl_h(i).d.valid := rsp_valid && (host_index === i.U)
    io.tl_h(i).d.bits := rsp_bits
    io.tl_h(i).d.bits.source := rsp_bits.source >> StIdW
  }

  // Combine ready signals back to device D-channel
  io.tl_d.d.ready := VecInit(io.tl_h.map(_.d.ready))(host_index)
}

import _root_.circt.stage.{ChiselStage, FirtoolOption}
import chisel3.stage.ChiselGeneratorAnnotation
import scala.annotation.nowarn

@nowarn
object TlulSocketM1_2_128Emitter extends App {
  val p = new Parameters
  p.lsuDataBits = 128
  (new ChiselStage).execute(
    Array("--target", "systemverilog") ++ args,
    Seq(
      ChiselGeneratorAnnotation(() =>
        new TlulSocketM1(
          p = new bus.TLULParameters(p),
          M = 2,
          HReqDepth = Seq.fill(2)(0),
          HRspDepth = Seq.fill(2)(0),
          DReqDepth = 0,
          DRspDepth = 0,
          moduleName = "TlulSocketM1_2_128"
        )
      )
    ) ++ Seq(FirtoolOption("-enable-layers=Verification"))
  )
}

@nowarn
object TlulSocketM1_3_128Emitter extends App {
  val p = new Parameters
  p.lsuDataBits = 128
  (new ChiselStage).execute(
    Array("--target", "systemverilog") ++ args,
    Seq(
      ChiselGeneratorAnnotation(() =>
        new TlulSocketM1(
          p = new bus.TLULParameters(p),
          M = 3,
          HReqDepth = Seq.fill(3)(0),
          HRspDepth = Seq.fill(3)(0),
          DReqDepth = 0,
          DRspDepth = 0,
          moduleName = "TlulSocketM1_3_128"
        )
      )
    ) ++ Seq(FirtoolOption("-enable-layers=Verification"))
  )
}
