$(".btnPredict").on("click", function(e){
	 $.ajax("{{ url_for('/')}}").done(function(reply) {
	 	$("#predict").html(reply);
	 });
	 $(".hasil-predict").first().slideToggle("slow");
});
